from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

from model.frameworks.v4.tg import VariableClassTGVPR, fixed_class_folds, topology_loss
from model.candidates.v2.modules.ort import OrthogonalMix, orthogonal_transport, residual_subspace
from model.candidates.v2.trainers.train_elpt import FrozenPrototypeClassifier, _fold_package, _load_fold_checkpoint
from model.frameworks.v4.tst import TangentStepGate, tangent_transport
from model.frameworks.v2 import train as h1
from tools.reproducibility import configure_reproducibility
from tools.run_contract import atomic_write_json, current_code_commit, prepare_output_dir, require_clean_code_tree
from tools.runtime import sha256_file


CONFIG_KEYS={"schema_version","attempt_id","idea_id","framework_id","base_config","base_checkpoint","base_checkpoint_sha256","tst_gate_model","tst_gate_model_sha256","fold_checkpoint_dir","seed","epochs","batch_half","lr","weight_decay","topology_weight","subspace_rank","initial_mix","parent_metrics_percent"}
CONFIG_KEYS_V2=CONFIG_KEYS|{"projection_mode"}


class TeeStream:
    def __init__(self,*streams): self.streams=streams
    def write(self,value):
        for stream in self.streams: stream.write(value)
        return len(value)
    def flush(self):
        for stream in self.streams: stream.flush()


def load_config(path:Path):
    path=path.resolve(); config=yaml.safe_load(path.read_text(encoding="utf-8")); actual=set(config) if isinstance(config,dict) else set()
    expected=CONFIG_KEYS_V2 if isinstance(config,dict) and config.get("schema_version")=="gzsl-paper.ort.v2" else CONFIG_KEYS
    if not isinstance(config,dict) or actual!=expected: raise ValueError(f"ORT配置字段错误；缺少={sorted(expected-actual)}，多出={sorted(actual-expected)}。")
    if config["schema_version"] not in ("gzsl-paper.ort.v1","gzsl-paper.ort.v2") or config["attempt_id"] not in ("V2-TRY-056","V2-TRY-057") or config["idea_id"]!="IDEA-017": raise ValueError("ORT首次TRY身份错误。")
    if int(config["epochs"])!=20 or int(config["batch_half"])!=32 or float(config["lr"])!=0.01 or float(config["weight_decay"])!=0.0: raise ValueError("ORT训练参数错误。")
    if int(config["subspace_rank"])!=32 or float(config["initial_mix"])!=0.1 or float(config["topology_weight"])!=0.1: raise ValueError("ORT模块参数错误。")
    if set(config["parent_metrics_percent"])!={"U","S","H","ZS"}: raise ValueError("ORT父指标不完整。")
    config.setdefault("projection_mode","shared")
    expected_mode="shared" if config["attempt_id"]=="V2-TRY-056" else "complement"
    if config["projection_mode"]!=expected_mode: raise ValueError("ORT投影模式与TRY身份错误。")
    return config,sha256_file(path)


def _load_gate(config,device):
    path=Path(config["tst_gate_model"])
    if sha256_file(path)!=config["tst_gate_model_sha256"]: raise ValueError("ORT父TST gate SHA不匹配。")
    payload=torch.load(path,map_location="cpu",weights_only=False); gate=TangentStepGate(input_dim=4,max_step=1.5); gate.load_state_dict(payload["gate_state_dict"],strict=True)
    for p in gate.parameters(): p.requires_grad_(False)
    return gate.to(device).eval()


def run(config_path:Path,output_dir:Path,expected_commit:str):
    require_clean_code_tree(); code_commit=current_code_commit()
    if code_commit!=expected_commit: raise ValueError("expected-commit与当前HEAD不一致。")
    config,config_sha=load_config(config_path); base_path=Path(config["base_config"])
    if not base_path.is_absolute(): base_path=Path.cwd()/base_path
    base_config,base_config_sha=h1.load_config(base_path); paths=h1.resolve_paths(base_config); input_sha=h1.verify_inputs(base_config,paths,h1.TRAINING_KEYS); checkpoint_path=Path(config["base_checkpoint"])
    if sha256_file(checkpoint_path)!=config["base_checkpoint_sha256"]: raise ValueError("ORT父checkpoint SHA不匹配。")
    device=torch.device(base_config["device"])
    if device.type!="cuda" or not torch.cuda.is_available(): raise RuntimeError("ORT要求CUDA。")
    output_dir=prepare_output_dir(output_dir)
    with (output_dir/"config.snapshot.yaml").open("x",encoding="utf-8") as handle: yaml.safe_dump(config,handle,allow_unicode=True,sort_keys=False)
    log_handle=(output_dir/"training.log").open("x",encoding="utf-8",buffering=1); original_stdout=sys.stdout; sys.stdout=TeeStream(sys.stdout,log_handle)
    try:
        seed=int(config["seed"]); configure_reproducibility(seed,strict_determinism=True,deterministic_warn_only=False); tensors={name:torch.load(paths[name],map_location="cpu",weights_only=True) for name in ("sentence_embeds","train_features","train_labels")}; labels=tensors["train_labels"].long(); seenclasses=torch.unique(labels,sorted=True); allclasses=torch.arange(200); unseenclasses=allclasses[~torch.isin(allclasses,seenclasses)]; folds=fixed_class_folds(seenclasses); gate=_load_gate(config,device); packages=[]
        for fold_id,(ps,pu) in enumerate(folds):
            fold_model=_load_fold_checkpoint(fold_id,ps,tensors["sentence_embeds"],tensors["train_features"],labels,base_config,device,config["fold_checkpoint_dir"]); package=_fold_package(fold_model,ps,pu,tensors,seenclasses,device,"summary")
            with torch.no_grad():
                base_all=package["base_all"].to(device); fold_full=package["fold_full"].to(device); value=package["value"].to(device); step=gate(package["gate_features"].to(device)); basis=residual_subspace(base_all,fold_full.index_select(0,ps.to(device)),ps,config["subspace_rank"]); package["basis"]=basis; package["value_gpu"]=value; package["step_gpu"]=step
            packages.append(package); del fold_model
        mix=OrthogonalMix(config["initial_mix"]).to(device); optimizer=torch.optim.Adam(mix.parameters(),lr=float(config["lr"]),weight_decay=0.0); mapping=torch.full((200,),-1,dtype=torch.long); mapping[seenclasses]=torch.arange(150); generators=[torch.Generator(device="cpu").manual_seed(seed*17000+i) for i in range(3)]; half=int(config["batch_half"]); history=[]
        for epoch in range(1,int(config["epochs"])+1):
            loss_sum=0.0; count=0
            for fold_id,package in enumerate(packages):
                steps=min(package["seen_indices"].numel()//half,package["unseen_indices"].numel()//half)
                for _ in range(steps):
                    g=generators[fold_id]; si=package["seen_indices"][torch.randperm(package["seen_indices"].numel(),generator=g)[:half]]; ui=package["unseen_indices"][torch.randperm(package["unseen_indices"].numel(),generator=g)[:half]]; indices=torch.cat((si,ui)); images=tensors["train_features"][indices].to(device).float(); targets=mapping[labels[indices]].to(device); final=package["fold_full"].to(device).clone(); pu=package["pseudo_unseen"].to(device); base_target=package["base_all"].to(device).index_select(0,pu); final[pu]=orthogonal_transport(base_target,package["value_gpu"],package["step_gpu"],package["basis"],mix(),config["projection_mode"]); competition=final.index_select(0,seenclasses.to(device)); logits=F.normalize(images,dim=-1)@competition.T*package["scale"].to(device); ce=F.cross_entropy(logits,targets); topo=topology_loss(package["base_all"].to(device).index_select(0,seenclasses.to(device)),competition); loss=ce+float(config["topology_weight"])*topo; optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step(); loss_sum+=float(loss.detach()); count+=1
            row={"epoch":epoch,"loss":loss_sum/count,"mix":float(mix().detach())}; history.append(row); print(f"epoch={epoch} loss={row['loss']:.6f} mix={row['mix']:.6f}")
        torch.save({"attempt_id":config["attempt_id"],"code_commit":code_commit,"config":config,"mix_state_dict":copy.deepcopy(mix.state_dict()),"history":history},output_dir/"ort_model.pth")
        # official test严格在ORT训练结束后加载。
        input_sha.update(h1.verify_inputs(base_config,paths,h1.OFFICIAL_KEYS)); tensors.update({name:torch.load(paths[name],map_location="cpu",weights_only=True) for name in h1.OFFICIAL_KEYS}); checkpoint=torch.load(checkpoint_path,map_location="cpu",weights_only=False); centroids=h1.visual_centroids(tensors["train_features"],labels,seenclasses); parent=VariableClassTGVPR(tensors["sentence_embeds"],seenclasses,centroids,dropout=base_config["dropout"],inner_ratio=base_config["inner_ratio"],outer_ratio=base_config["outer_ratio"],temperature=base_config["temperature"]); parent.load_state_dict(checkpoint["model_state_dict"],strict=True); parent=parent.to(device).eval(); base_all=parent.base_prototypes(); full=parent.prototypes().clone(); value=parent.value_candidate(unseenclasses.to(device)); base_target=base_all.index_select(0,unseenclasses.to(device)); features=torch.stack(((base_target*value).sum(dim=-1),(value-base_target).norm(dim=-1),(base_target@base_all.index_select(0,seenclasses.to(device)).T).topk(5,dim=1).values.mean(dim=1),(base_target@base_all.index_select(0,seenclasses.to(device)).T).topk(5,dim=1).values.max(dim=1).values),dim=1); step=gate(features); basis=residual_subspace(base_all,full.index_select(0,seenclasses.to(device)),seenclasses,config["subspace_rank"]); tst_full=full.clone(); tst_full[unseenclasses.to(device)]=tangent_transport(base_target,value,step); candidate=full.clone(); candidate[unseenclasses.to(device)]=orthogonal_transport(base_target,value,step,basis,mix(),config["projection_mode"]); parent_metrics=h1.evaluate(FrozenPrototypeClassifier(tst_full,parent.scale()).to(device),tensors,seenclasses,unseenclasses,device); candidate_metrics=h1.evaluate(FrozenPrototypeClassifier(candidate,parent.scale()).to(device),tensors,seenclasses,unseenclasses,device); delta={key:candidate_metrics[key]-float(config["parent_metrics_percent"][key]) for key in ("U","S","H","ZS")}; value_mix=float(mix().detach()); success=delta["H"]>=0.20 and delta["U"]>=-2 and delta["S"]>=-2 and value_mix<0.98; atomic_write_json(output_dir/"data_fingerprints.json",{"files":input_sha}); metrics={"attempt_id":config["attempt_id"],"idea_id":config["idea_id"],"framework_id":config["framework_id"],"code_commit":code_commit,"config_sha256":config_sha,"base_config_sha256":base_config_sha,"evaluation_protocol":h1.EVALUATION_PROTOCOL,"test_used_for_selection":True,"unseen_images_used_for_gradient":False,"recomputed_parent_metrics_percent":parent_metrics,"parent_metrics_percent":config["parent_metrics_percent"],"candidate_metrics_percent":candidate_metrics,"delta_vs_parent_percent_points":delta,"learned_mix":value_mix,"projection_mode":config["projection_mode"],"success":success,"ort_model_sha256":sha256_file(output_dir/"ort_model.pth")}; atomic_write_json(output_dir/"metrics.json",metrics); print(metrics); return metrics
    finally:
        sys.stdout.flush(); sys.stdout=original_stdout; log_handle.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--config",type=Path,required=True); parser.add_argument("--output-dir",type=Path,required=True); parser.add_argument("--expected-commit",required=True); args=parser.parse_args(); run(args.config,args.output_dir,args.expected_commit)


if __name__=="__main__": main()
