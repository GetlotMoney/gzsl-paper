from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

from model.innovations.ccgr import ClassConditionedGeometricGenerator, tangent_direction_basis
from model.innovations.cgfg import ConditionalGaussianFeatureGenerator, CosineLinearClassifier
from model.innovations.dpt import text_resultant_lengths
from model.innovations.elpt import VariableClassTGVPR, fixed_class_folds, topology_loss
from model.innovations.train_elpt import _candidate_prototypes
from model.innovations.tst import TangentStepGate
from model.tg_vpr_h1 import train as h1
from tools.reproducibility import configure_reproducibility
from tools.run_contract import atomic_write_json, current_code_commit, prepare_output_dir, require_clean_code_tree
from tools.runtime import sha256_file


CONFIG_KEYS={"schema_version","attempt_id","idea_id","framework_id","base_config","base_checkpoint","base_checkpoint_sha256","ntr_gate_model","ntr_gate_model_sha256","ccgr_model","ccgr_model_sha256","seed","generator_epochs","classifier_epochs","batch_half","generator_lr","classifier_lr","weight_decay","hidden_dim","max_residual_norm","synthetic_per_class","anchor_weight","parent_metrics_percent"}


class TeeStream:
    def __init__(self,*streams): self.streams=streams
    def write(self,value):
        for stream in self.streams: stream.write(value)
        return len(value)
    def flush(self):
        for stream in self.streams: stream.flush()


def load_config(path:Path):
    path=path.resolve(); config=yaml.safe_load(path.read_text(encoding="utf-8")); actual=set(config) if isinstance(config,dict) else set()
    if not isinstance(config,dict) or actual!=CONFIG_KEYS: raise ValueError(f"CGFG配置字段错误；缺少={sorted(CONFIG_KEYS-actual)}，多出={sorted(actual-CONFIG_KEYS)}。")
    if config["schema_version"]!="gzsl-paper.cgfg.v1" or config["attempt_id"]!="V2-TRY-076" or config["idea_id"]!="IDEA-024": raise ValueError("CGFG首次TRY身份错误。")
    if int(config["generator_epochs"])!=200 or int(config["classifier_epochs"])!=20 or int(config["batch_half"])!=64: raise ValueError("CGFG训练轮次错误。")
    if float(config["generator_lr"])!=0.001 or float(config["classifier_lr"])!=0.001 or float(config["weight_decay"])!=0.0001: raise ValueError("CGFG优化器错误。")
    if int(config["hidden_dim"])!=128 or float(config["max_residual_norm"])!=0.2 or int(config["synthetic_per_class"])!=300 or float(config["anchor_weight"])!=0.1: raise ValueError("CGFG模块参数错误。")
    if set(config["parent_metrics_percent"])!={"U","S","H","ZS"}: raise ValueError("CGFG父指标不完整。")
    return config,sha256_file(path)


def _load_gate(config,device):
    path=Path(config["ntr_gate_model"])
    if sha256_file(path)!=config["ntr_gate_model_sha256"]: raise ValueError("CGFG父NTR gate SHA不匹配。")
    payload=torch.load(path,map_location="cpu",weights_only=False); gate=TangentStepGate(input_dim=8,max_step=1.5); gate.load_state_dict(payload["gate_state_dict"],strict=True)
    for p in gate.parameters(): p.requires_grad_(False)
    return gate.to(device).eval()


@torch.no_grad()
def evaluate(classifier,tensors,seenclasses,unseenclasses,device):
    def pred(features,class_ids=None):
        result=classifier.logits(features.to(device).float(),class_ids).argmax(1).cpu(); return class_ids[result] if class_ids is not None else result
    sp=pred(tensors["seen_features"]); up=pred(tensors["unseen_features"]); zp=pred(tensors["unseen_features"],unseenclasses); s=h1.per_class_accuracy(tensors["seen_labels"],sp,seenclasses); u=h1.per_class_accuracy(tensors["unseen_labels"],up,unseenclasses); z=h1.per_class_accuracy(tensors["unseen_labels"],zp,unseenclasses); H=2*s*u/(s+u) if s+u else 0.0; return {"U":u*100,"S":s*100,"H":H*100,"ZS":z*100}


def run(config_path:Path,output_dir:Path,expected_commit:str):
    require_clean_code_tree(); code_commit=current_code_commit()
    if code_commit!=expected_commit: raise ValueError("expected-commit与当前HEAD不一致。")
    config,config_sha=load_config(config_path); base_path=Path(config["base_config"])
    if not base_path.is_absolute(): base_path=Path.cwd()/base_path
    base_config,base_config_sha=h1.load_config(base_path); paths=h1.resolve_paths(base_config); input_sha=h1.verify_inputs(base_config,paths,h1.TRAINING_KEYS); checkpoint_path=Path(config["base_checkpoint"])
    if sha256_file(checkpoint_path)!=config["base_checkpoint_sha256"] or sha256_file(Path(config["ccgr_model"]))!=config["ccgr_model_sha256"]: raise ValueError("CGFG父模型SHA不匹配。")
    device=torch.device(base_config["device"])
    if device.type!="cuda" or not torch.cuda.is_available(): raise RuntimeError("CGFG要求CUDA。")
    output_dir=prepare_output_dir(output_dir)
    with (output_dir/"config.snapshot.yaml").open("x",encoding="utf-8") as handle: yaml.safe_dump(config,handle,allow_unicode=True,sort_keys=False)
    log_handle=(output_dir/"training.log").open("x",encoding="utf-8",buffering=1); original_stdout=sys.stdout; sys.stdout=TeeStream(sys.stdout,log_handle)
    try:
        seed=int(config["seed"]); configure_reproducibility(seed,strict_determinism=True,deterministic_warn_only=False); tensors={name:torch.load(paths[name],map_location="cpu",weights_only=True) for name in ("sentence_embeds","train_features","train_labels")}; labels=tensors["train_labels"].long(); seenclasses=torch.unique(labels,sorted=True); allclasses=torch.arange(200); unseenclasses=allclasses[~torch.isin(allclasses,seenclasses)]; checkpoint=torch.load(checkpoint_path,map_location="cpu",weights_only=False); centroids=h1.visual_centroids(tensors["train_features"],labels,seenclasses); parent=VariableClassTGVPR(tensors["sentence_embeds"],seenclasses,centroids,dropout=base_config["dropout"],inner_ratio=base_config["inner_ratio"],outer_ratio=base_config["outer_ratio"],temperature=base_config["temperature"]); parent.load_state_dict(checkpoint["model_state_dict"],strict=True); parent=parent.to(device).eval(); gate=_load_gate(config,device); ntr,_=_candidate_prototypes(parent,gate,seenclasses,unseenclasses,device,"top5_vector",fixed_class_folds(seenclasses),"tangent"); base=parent.base_prototypes(); value=parent.value_candidate(allclasses.to(device)); roles=parent.semantic_group_vectors(); basis=tangent_direction_basis(base,value,roles); sim=base@base.index_select(0,seenclasses.to(device)).T; top5=sim.topk(5,dim=1).values; features=torch.stack(((base*value).sum(-1),(value-base).norm(dim=-1),text_resultant_lengths(tensors["sentence_embeds"]).to(device),top5.mean(1)),dim=1); ccgr=ClassConditionedGeometricGenerator(ntr,basis,features,unseenclasses,parent.scale(),hidden_dim=32,max_magnitude=0.2,initial_magnitude=0.02).to(device); cp=torch.load(Path(config["ccgr_model"]),map_location="cpu",weights_only=False); ccgr.load_state_dict(cp["model_state_dict"],strict=True); ccgr.eval(); semantic=ccgr.prototypes().detach(); generator=ConditionalGaussianFeatureGenerator(semantic,config["hidden_dim"],config["max_residual_norm"]).to(device); gen_opt=torch.optim.Adam(generator.parameters(),lr=float(config["generator_lr"]),weight_decay=float(config["weight_decay"])); gen_history=[]
        for epoch in range(1,int(config["generator_epochs"])+1):
            means=generator.means(); seen_means=means.index_select(0,seenclasses.to(device)); alignment=1.0-(seen_means*centroids.to(device)).sum(-1).mean(); topo=topology_loss(semantic,means); loss=alignment+0.1*topo; gen_opt.zero_grad(set_to_none=True); loss.backward(); gen_opt.step(); row={"epoch":epoch,"alignment":float(alignment.detach()),"topology":float(topo.detach()),"residual_max":float(generator.residual().norm(dim=-1).max().detach())}; gen_history.append(row)
            if epoch in (1,10,20,50,100,150,200): print(f"gen_epoch={epoch} alignment={row['alignment']:.6f} topology={row['topology']:.6f}")
        with torch.no_grad():
            normalized_train=F.normalize(tensors["train_features"].to(device).float(),dim=-1); global_to_seen=torch.full((200,),-1,dtype=torch.long); global_to_seen[seenclasses]=torch.arange(150); local_labels=global_to_seen[labels].to(device); residual_bank=normalized_train-centroids.to(device).index_select(0,local_labels); means=generator.means(); synth_features=[]; synth_labels=[]; rng=torch.Generator(device="cpu").manual_seed(seed*31)
            for class_id in unseenclasses:
                idx=torch.randint(residual_bank.size(0),(int(config["synthetic_per_class"]),),generator=rng); noise=residual_bank.index_select(0,idx.to(device)); mu=means[int(class_id)].unsqueeze(0); synth_features.append(F.normalize(mu+noise,dim=-1)); synth_labels.append(torch.full((int(config["synthetic_per_class"]),),int(class_id),dtype=torch.long,device=device))
            synth_features=torch.cat(synth_features); synth_labels=torch.cat(synth_labels); initial=semantic.clone(); initial[unseenclasses.to(device)]=means.index_select(0,unseenclasses.to(device)); classifier=CosineLinearClassifier(initial,parent.scale()).to(device); anchor=F.normalize(initial.detach(),dim=-1)
        cls_opt=torch.optim.Adam(classifier.parameters(),lr=float(config["classifier_lr"]),weight_decay=float(config["weight_decay"])); cls_history=[]; half=int(config["batch_half"]); seen_indices=torch.arange(normalized_train.size(0)); unseen_indices=torch.arange(synth_features.size(0)); seen_rng=torch.Generator(device="cpu").manual_seed(seed*37); unseen_rng=torch.Generator(device="cpu").manual_seed(seed*41)
        for epoch in range(1,int(config["classifier_epochs"])+1):
            loss_sum=0.0; count=0; steps=max(normalized_train.size(0),synth_features.size(0))//half
            for _ in range(steps):
                si=seen_indices[torch.randperm(seen_indices.numel(),generator=seen_rng)[:half]].to(device); ui=unseen_indices[torch.randperm(unseen_indices.numel(),generator=unseen_rng)[:half]].to(device); x=torch.cat((normalized_train.index_select(0,si),synth_features.index_select(0,ui))); y=torch.cat((labels.index_select(0,si.cpu()).to(device),synth_labels.index_select(0,ui))); ce=F.cross_entropy(classifier.logits(x),y); anchor_loss=1.0-(F.normalize(classifier.weight,dim=-1)*anchor).sum(-1).mean(); loss=ce+float(config["anchor_weight"])*anchor_loss; cls_opt.zero_grad(set_to_none=True); loss.backward(); cls_opt.step(); loss_sum+=float(loss.detach()); count+=1
            row={"epoch":epoch,"loss":loss_sum/count}; cls_history.append(row); print(f"cls_epoch={epoch} loss={row['loss']:.6f}")
        torch.save({"attempt_id":config["attempt_id"],"code_commit":code_commit,"config":config,"generator_state_dict":copy.deepcopy(generator.state_dict()),"classifier_state_dict":copy.deepcopy(classifier.state_dict()),"generator_history":gen_history,"classifier_history":cls_history},output_dir/"cgfg_model.pth")
        # official test严格在CGFG训练结束后加载。
        input_sha.update(h1.verify_inputs(base_config,paths,h1.OFFICIAL_KEYS)); tensors.update({name:torch.load(paths[name],map_location="cpu",weights_only=True) for name in h1.OFFICIAL_KEYS}); parent_classifier=CosineLinearClassifier(semantic,parent.scale()).to(device); parent_metrics=evaluate(parent_classifier,tensors,seenclasses,unseenclasses,device); candidate_metrics=evaluate(classifier,tensors,seenclasses,unseenclasses,device); delta={key:candidate_metrics[key]-float(config["parent_metrics_percent"][key]) for key in ("U","S","H","ZS")}; success=delta["H"]>=0.20 and delta["U"]>=-2 and delta["S"]>=-2; atomic_write_json(output_dir/"data_fingerprints.json",{"files":input_sha}); metrics={"attempt_id":config["attempt_id"],"idea_id":config["idea_id"],"framework_id":config["framework_id"],"code_commit":code_commit,"config_sha256":config_sha,"base_config_sha256":base_config_sha,"evaluation_protocol":h1.EVALUATION_PROTOCOL,"test_used_for_selection":True,"unseen_images_used_for_gradient":False,"recomputed_parent_metrics_percent":parent_metrics,"parent_metrics_percent":config["parent_metrics_percent"],"candidate_metrics_percent":candidate_metrics,"delta_vs_parent_percent_points":delta,"success":success,"cgfg_model_sha256":sha256_file(output_dir/"cgfg_model.pth")}; atomic_write_json(output_dir/"metrics.json",metrics); print(metrics); return metrics
    finally:
        sys.stdout.flush(); sys.stdout=original_stdout; log_handle.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--config",type=Path,required=True); parser.add_argument("--output-dir",type=Path,required=True); parser.add_argument("--expected-commit",required=True); args=parser.parse_args(); run(args.config,args.output_dir,args.expected_commit)


if __name__=="__main__": main()
