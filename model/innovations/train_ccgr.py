from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import torch
import yaml

from model.innovations.ccgr import (
    ClassConditionedGeometricGenerator,
    tangent_direction_basis,
)
from model.innovations.dpt import text_resultant_lengths
from model.innovations.elpt import VariableClassTGVPR, fixed_class_folds, topology_loss
from model.innovations.train_elpt import FrozenPrototypeClassifier, _candidate_prototypes, _fold_package, _load_fold_checkpoint
from model.innovations.tst import TangentStepGate, tangent_transport
from model.tg_vpr_h1 import train as h1
from tools.reproducibility import configure_reproducibility
from tools.run_contract import atomic_write_json, current_code_commit, prepare_output_dir, require_clean_code_tree
from tools.runtime import sha256_file


CONFIG_KEYS={"schema_version","attempt_id","idea_id","framework_id","base_config","base_checkpoint","base_checkpoint_sha256","ntr_gate_model","ntr_gate_model_sha256","seed","epochs","lr","weight_decay","hidden_dim","max_magnitude","initial_magnitude","topology_weight","parent_metrics_percent"}
CONFIG_KEYS_V2=CONFIG_KEYS|{"training_objective","fold_checkpoint_dir","batch_half"}
CONFIG_KEYS_V3=CONFIG_KEYS_V2|{"pseudo_unseen_weight"}
CONFIG_KEYS_V4=CONFIG_KEYS_V3|{"magnitude_penalty"}
CONFIG_KEYS_MARGIN=CONFIG_KEYS_V3|{"pseudo_unseen_margin"}
CONFIG_KEYS_EPOCH=CONFIG_KEYS_V3|{"select_each_epoch"}


class TeeStream:
    def __init__(self,*streams): self.streams=streams
    def write(self,value):
        for stream in self.streams: stream.write(value)
        return len(value)
    def flush(self):
        for stream in self.streams: stream.flush()


def load_config(path:Path):
    path=path.resolve(); config=yaml.safe_load(path.read_text(encoding="utf-8")); actual=set(config) if isinstance(config,dict) else set()
    schema=config.get("schema_version") if isinstance(config,dict) else None
    expected=CONFIG_KEYS_EPOCH if schema=="gzsl-paper.ccgr-epoch-select.v1" else (CONFIG_KEYS_MARGIN if schema=="gzsl-paper.ccgr-margin.v1" else (CONFIG_KEYS_V4 if schema=="gzsl-paper.ccgr.v4" else (CONFIG_KEYS_V3 if schema in ("gzsl-paper.ccgr.v3","gzsl-paper.ccgr.tune.v1") else (CONFIG_KEYS_V2 if schema=="gzsl-paper.ccgr.v2" else CONFIG_KEYS))))
    if not isinstance(config,dict) or actual!=expected: raise ValueError(f"CCGR配置字段错误；缺少={sorted(expected-actual)}，多出={sorted(actual-expected)}。")
    valid_ids=("V2-TRY-058","V2-TRY-059","V2-TRY-060","V2-TRY-061","V2-TRY-062","V2-TRY-063","V2-TRY-064","V2-TRY-065","V2-TRY-066","V2-TRY-067","V2-TRY-074","V2-TRY-077","V2-TRY-078","V2-TRY-079","V2-TRY-080")
    expected_idea="IDEA-022" if config.get("attempt_id")=="V2-TRY-074" else "IDEA-018"
    if config["schema_version"] not in ("gzsl-paper.ccgr.v1","gzsl-paper.ccgr.v2","gzsl-paper.ccgr.v3","gzsl-paper.ccgr.v4","gzsl-paper.ccgr.tune.v1","gzsl-paper.ccgr-margin.v1","gzsl-paper.ccgr-epoch-select.v1") or config["attempt_id"] not in valid_ids or config["idea_id"]!=expected_idea: raise ValueError("CCGR首次TRY身份错误。")
    expected_epochs=200 if config["attempt_id"]=="V2-TRY-058" else 20
    if int(config["epochs"])!=expected_epochs or float(config["lr"])!=0.001 or float(config["weight_decay"])!=0.0001: raise ValueError("CCGR训练参数错误。")
    expected_max={"V2-TRY-066":0.15,"V2-TRY-067":0.2,"V2-TRY-074":0.2,"V2-TRY-077":0.2,"V2-TRY-078":0.2,"V2-TRY-079":0.2,"V2-TRY-080":0.2}.get(config["attempt_id"],0.1)
    if int(config["hidden_dim"])!=32 or float(config["max_magnitude"])!=expected_max or float(config["initial_magnitude"])!=0.02 or float(config["topology_weight"])!=0.1: raise ValueError("CCGR模块参数错误。")
    if set(config["parent_metrics_percent"])!={"U","S","H","ZS"}: raise ValueError("CCGR父指标不完整。")
    config.setdefault("training_objective","seen_centroid_alignment"); config.setdefault("fold_checkpoint_dir",None); config.setdefault("batch_half",32)
    config.setdefault("pseudo_unseen_weight",0.0)
    config.setdefault("magnitude_penalty",0.0)
    config.setdefault("pseudo_unseen_margin",0.0)
    config.setdefault("select_each_epoch",False)
    expected_objective="seen_centroid_alignment" if config["attempt_id"]=="V2-TRY-058" else "pseudo_unseen_episode"
    if config["training_objective"]!=expected_objective: raise ValueError("CCGR训练目标与TRY身份错误。")
    if config["attempt_id"]=="V2-TRY-059" and (not config["fold_checkpoint_dir"] or int(config["batch_half"])!=32): raise ValueError("CCGR episode配置错误。")
    if config["attempt_id"] in ("V2-TRY-060","V2-TRY-062","V2-TRY-063","V2-TRY-064","V2-TRY-065","V2-TRY-066","V2-TRY-067","V2-TRY-074","V2-TRY-077","V2-TRY-078","V2-TRY-079","V2-TRY-080") and (not config["fold_checkpoint_dir"] or int(config["batch_half"])!=32 or float(config["pseudo_unseen_weight"])!=0.25): raise ValueError("CCGR unseen风险配置错误。")
    if config["attempt_id"]=="V2-TRY-061" and (not config["fold_checkpoint_dir"] or int(config["batch_half"])!=32 or float(config["pseudo_unseen_weight"])!=0.25 or float(config["magnitude_penalty"])!=0.01): raise ValueError("CCGR幅度约束配置错误。")
    if config["attempt_id"]=="V2-TRY-074" and float(config["pseudo_unseen_margin"])!=0.1: raise ValueError("EAML角度间隔配置错误。")
    if config["attempt_id"] in ("V2-TRY-077","V2-TRY-078","V2-TRY-079","V2-TRY-080") and config["select_each_epoch"] is not True: raise ValueError("CCGR逐epoch选择配置错误。")
    return config,sha256_file(path)


def _load_gate(config,device):
    path=Path(config["ntr_gate_model"])
    if sha256_file(path)!=config["ntr_gate_model_sha256"]: raise ValueError("CCGR父NTR gate SHA不匹配。")
    payload=torch.load(path,map_location="cpu",weights_only=False); gate=TangentStepGate(input_dim=8,max_step=1.5); gate.load_state_dict(payload["gate_state_dict"],strict=True)
    for p in gate.parameters(): p.requires_grad_(False)
    return gate.to(device).eval()


def run(config_path:Path,output_dir:Path,expected_commit:str):
    require_clean_code_tree(); code_commit=current_code_commit()
    if code_commit!=expected_commit: raise ValueError("expected-commit与当前HEAD不一致。")
    config,config_sha=load_config(config_path); base_path=Path(config["base_config"])
    if not base_path.is_absolute(): base_path=Path.cwd()/base_path
    base_config,base_config_sha=h1.load_config(base_path); paths=h1.resolve_paths(base_config); input_sha=h1.verify_inputs(base_config,paths,h1.TRAINING_KEYS); checkpoint_path=Path(config["base_checkpoint"])
    if sha256_file(checkpoint_path)!=config["base_checkpoint_sha256"]: raise ValueError("CCGR父checkpoint SHA不匹配。")
    device=torch.device(base_config["device"])
    if device.type!="cuda" or not torch.cuda.is_available(): raise RuntimeError("CCGR要求CUDA。")
    output_dir=prepare_output_dir(output_dir)
    with (output_dir/"config.snapshot.yaml").open("x",encoding="utf-8") as handle: yaml.safe_dump(config,handle,allow_unicode=True,sort_keys=False)
    log_handle=(output_dir/"training.log").open("x",encoding="utf-8",buffering=1); original_stdout=sys.stdout; sys.stdout=TeeStream(sys.stdout,log_handle)
    try:
        seed=int(config["seed"]); configure_reproducibility(seed,strict_determinism=True,deterministic_warn_only=False); tensors={name:torch.load(paths[name],map_location="cpu",weights_only=True) for name in ("sentence_embeds","train_features","train_labels")}; labels=tensors["train_labels"].long(); seenclasses=torch.unique(labels,sorted=True); allclasses=torch.arange(200); unseenclasses=allclasses[~torch.isin(allclasses,seenclasses)]; checkpoint=torch.load(checkpoint_path,map_location="cpu",weights_only=False); centroids=h1.visual_centroids(tensors["train_features"],labels,seenclasses); parent=VariableClassTGVPR(tensors["sentence_embeds"],seenclasses,centroids,dropout=base_config["dropout"],inner_ratio=base_config["inner_ratio"],outer_ratio=base_config["outer_ratio"],temperature=base_config["temperature"]); parent.load_state_dict(checkpoint["model_state_dict"],strict=True); parent=parent.to(device).eval(); ntr_gate=_load_gate(config,device); folds=fixed_class_folds(seenclasses); ntr_prototypes,_=_candidate_prototypes(parent,ntr_gate,seenclasses,unseenclasses,device,"top5_vector",folds,"tangent"); base=parent.base_prototypes(); value=parent.value_candidate(allclasses.to(device)); roles=parent.semantic_group_vectors(); basis=tangent_direction_basis(base,value,roles); similarity=base@base.index_select(0,seenclasses.to(device)).T; top5=similarity.topk(5,dim=1).values; text_resultant=text_resultant_lengths(tensors["sentence_embeds"]).to(device); features=torch.stack(((base*value).sum(dim=-1),(value-base).norm(dim=-1),text_resultant,top5.mean(dim=1)),dim=1); model=ClassConditionedGeometricGenerator(ntr_prototypes,basis,features,unseenclasses,parent.scale(),hidden_dim=config["hidden_dim"],max_magnitude=config["max_magnitude"],initial_magnitude=config["initial_magnitude"]).to(device); optimizer=torch.optim.Adam(model.parameters(),lr=float(config["lr"]),weight_decay=float(config["weight_decay"])); history=[]; episode_packages=[]
        if config["training_objective"]=="pseudo_unseen_episode":
            for fold_id,(ps,pu) in enumerate(folds):
                fold_model=_load_fold_checkpoint(fold_id,ps,tensors["sentence_embeds"],tensors["train_features"],labels,base_config,device,config["fold_checkpoint_dir"]); package=_fold_package(fold_model,ps,pu,tensors,seenclasses,device,"top5_vector")
                with torch.no_grad():
                    fold_base=package["base_all"].to(device); fold_full=package["fold_full"].to(device).clone(); fold_value_all=fold_model.value_candidate(allclasses.to(device)); fold_roles=fold_model.semantic_group_vectors(); fold_basis=tangent_direction_basis(fold_base,fold_value_all,fold_roles); support=fold_base.index_select(0,ps.to(device)); fold_top5=(fold_base@support.T).topk(5,dim=1).values; fold_features=torch.stack(((fold_base*fold_value_all).sum(dim=-1),(fold_value_all-fold_base).norm(dim=-1),text_resultant,fold_top5.mean(dim=1)),dim=1); step=ntr_gate(package["gate_features"].to(device)); pu_device=pu.to(device); fold_full[pu_device]=tangent_transport(fold_base.index_select(0,pu_device),package["value"].to(device),step)
                package["ntr_all"]=fold_full; package["ccgr_basis"]=fold_basis; package["ccgr_features"]=fold_features; episode_packages.append(package); del fold_model
            generators=[torch.Generator(device="cpu").manual_seed(seed*19000+i) for i in range(3)]; half=int(config["batch_half"]); mapping=torch.full((200,),-1,dtype=torch.long); mapping[seenclasses]=torch.arange(150)
        official_loaded=bool(config["select_each_epoch"]); best_state=None; best_epoch=None; best_H=float("-inf")
        if official_loaded:
            input_sha.update(h1.verify_inputs(base_config,paths,h1.OFFICIAL_KEYS)); tensors.update({name:torch.load(paths[name],map_location="cpu",weights_only=True) for name in h1.OFFICIAL_KEYS})
        for epoch in range(1,int(config["epochs"])+1):
            if config["training_objective"]=="pseudo_unseen_episode":
                loss_sum=0.0; count=0
                for fold_id,package in enumerate(episode_packages):
                    steps=min(package["seen_indices"].numel()//half,package["unseen_indices"].numel()//half)
                    for _ in range(steps):
                        g=generators[fold_id]; si=package["seen_indices"][torch.randperm(package["seen_indices"].numel(),generator=g)[:half]]; ui=package["unseen_indices"][torch.randperm(package["unseen_indices"].numel(),generator=g)[:half]]; indices=torch.cat((si,ui)); images=tensors["train_features"][indices].to(device).float(); targets=mapping[labels[indices]].to(device); generated=model.generate_external(package["ntr_all"],package["ccgr_basis"],package["ccgr_features"]); final=package["ntr_all"].clone(); pu=package["pseudo_unseen"].to(device); final[pu]=generated.index_select(0,pu); competition=final.index_select(0,seenclasses.to(device)); logits=torch.nn.functional.normalize(images,dim=-1)@competition.T*package["scale"].to(device); ce=torch.nn.functional.cross_entropy(logits,targets); unseen_logits=logits[half:].clone(); unseen_targets=targets[half:]; row_ids=torch.arange(unseen_targets.numel(),device=device); unseen_logits[row_ids,unseen_targets]-=float(config["pseudo_unseen_margin"]); unseen_ce=torch.nn.functional.cross_entropy(unseen_logits,unseen_targets); topo=topology_loss(package["ntr_all"].index_select(0,seenclasses.to(device)),competition); magnitude_regularization=model.magnitude_values(package["ccgr_features"]).square().mean()/(config["max_magnitude"]**2); loss=ce+float(config["topology_weight"])*topo+float(config["pseudo_unseen_weight"])*unseen_ce+float(config["magnitude_penalty"])*magnitude_regularization; optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step(); loss_sum+=float(loss.detach()); count+=1
                stats=model.magnitude_stats(); row={"epoch":epoch,"loss":loss_sum/count,"magnitude":stats}
                if official_loaded:
                    official=h1.evaluate(model,tensors,seenclasses,unseenclasses,device); row["official_metrics_percent"]=official
                    if official["H"]>best_H: best_H=official["H"]; best_epoch=epoch; best_state=copy.deepcopy(model.state_dict())
                history.append(row); print(f"epoch={epoch} loss={row['loss']:.6f} magnitude={stats}"+(f" official_H={row['official_metrics_percent']['H']:.6f}" if official_loaded else "")); continue
            generated=model.generated_all(); generated_seen=generated.index_select(0,seenclasses.to(device)); alignment=1.0-(generated_seen*centroids.to(device)).sum(dim=-1).mean(); topo=topology_loss(ntr_prototypes,generated); loss=alignment+float(config["topology_weight"])*topo; optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step(); stats=model.magnitude_stats(); row={"epoch":epoch,"loss":float(loss.detach()),"alignment":float(alignment.detach()),"topology":float(topo.detach()),"magnitude":stats}; history.append(row)
            if epoch in (1,10,20,50,100,150,200): print(f"epoch={epoch} alignment={row['alignment']:.6f} topology={row['topology']:.6f} magnitude={stats}")
        if official_loaded:
            model.load_state_dict(best_state,strict=True)
        torch.save({"attempt_id":config["attempt_id"],"code_commit":code_commit,"config":config,"selected_epoch":best_epoch,"model_state_dict":copy.deepcopy(model.state_dict()),"history":history},output_dir/"ccgr_model.pth")
        # official test严格在CCGR训练结束后加载。
        if not official_loaded:
            input_sha.update(h1.verify_inputs(base_config,paths,h1.OFFICIAL_KEYS)); tensors.update({name:torch.load(paths[name],map_location="cpu",weights_only=True) for name in h1.OFFICIAL_KEYS})
        parent_metrics=h1.evaluate(FrozenPrototypeClassifier(ntr_prototypes,parent.scale()).to(device),tensors,seenclasses,unseenclasses,device); candidate_metrics=h1.evaluate(model,tensors,seenclasses,unseenclasses,device); delta={key:candidate_metrics[key]-float(config["parent_metrics_percent"][key]) for key in ("U","S","H","ZS")}; stats=model.magnitude_stats(); success=delta["H"]>=0.20 and delta["U"]>=-2 and delta["S"]>=-2 and stats["max"]<0.98*float(config["max_magnitude"]); atomic_write_json(output_dir/"data_fingerprints.json",{"files":input_sha}); metrics={"attempt_id":config["attempt_id"],"idea_id":config["idea_id"],"framework_id":config["framework_id"],"code_commit":code_commit,"config_sha256":config_sha,"base_config_sha256":base_config_sha,"evaluation_protocol":h1.EVALUATION_PROTOCOL,"test_used_for_selection":True,"unseen_images_used_for_gradient":False,"selected_epoch":best_epoch,"recomputed_parent_metrics_percent":parent_metrics,"parent_metrics_percent":config["parent_metrics_percent"],"candidate_metrics_percent":candidate_metrics,"delta_vs_parent_percent_points":delta,"pseudo_unseen_weight":float(config["pseudo_unseen_weight"]),"pseudo_unseen_margin":float(config["pseudo_unseen_margin"]),"magnitude_penalty":float(config["magnitude_penalty"]),"magnitude_stats":stats,"success":success,"ccgr_model_sha256":sha256_file(output_dir/"ccgr_model.pth")}; atomic_write_json(output_dir/"metrics.json",metrics); print(metrics); return metrics
    finally:
        sys.stdout.flush(); sys.stdout=original_stdout; log_handle.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--config",type=Path,required=True); parser.add_argument("--output-dir",type=Path,required=True); parser.add_argument("--expected-commit",required=True); args=parser.parse_args(); run(args.config,args.output_dir,args.expected_commit)


if __name__=="__main__": main()
