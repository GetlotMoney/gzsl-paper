from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

from model.innovations.ccgr import ClassConditionedGeometricGenerator, tangent_direction_basis
from model.innovations.dpt import text_resultant_lengths
from model.innovations.elpt import VariableClassTGVPR, fixed_class_folds
from model.innovations.fvra import FeatureAdapterClassifier, FeatureVisualResidualAdapter
from model.innovations.train_elpt import _candidate_prototypes
from model.innovations.tst import TangentStepGate
from model.tg_vpr_h1 import train as h1
from tools.reproducibility import configure_reproducibility
from tools.run_contract import atomic_write_json, current_code_commit, prepare_output_dir, require_clean_code_tree
from tools.runtime import sha256_file


CONFIG_KEYS={"schema_version","attempt_id","idea_id","framework_id","base_config","base_checkpoint","base_checkpoint_sha256","ntr_gate_model","ntr_gate_model_sha256","ccgr_model","ccgr_model_sha256","seed","epochs","batch_size","lr","weight_decay","hidden_dim","residual_scale","consistency_weight","parent_metrics_percent"}


class TeeStream:
    def __init__(self,*streams): self.streams=streams
    def write(self,value):
        for stream in self.streams: stream.write(value)
        return len(value)
    def flush(self):
        for stream in self.streams: stream.flush()


def load_config(path:Path):
    path=path.resolve(); config=yaml.safe_load(path.read_text(encoding="utf-8")); actual=set(config) if isinstance(config,dict) else set()
    if not isinstance(config,dict) or actual!=CONFIG_KEYS: raise ValueError(f"FVRA配置字段错误；缺少={sorted(CONFIG_KEYS-actual)}，多出={sorted(actual-CONFIG_KEYS)}。")
    if config["schema_version"]!="gzsl-paper.fvra.v1" or config["attempt_id"]!="V2-TRY-068" or config["idea_id"]!="IDEA-019": raise ValueError("FVRA首次TRY身份错误。")
    if int(config["epochs"])!=20 or int(config["batch_size"])!=64 or float(config["lr"])!=0.001 or float(config["weight_decay"])!=0.0001: raise ValueError("FVRA训练参数错误。")
    if int(config["hidden_dim"])!=64 or float(config["residual_scale"])!=0.1 or float(config["consistency_weight"])!=0.1: raise ValueError("FVRA模块参数错误。")
    if set(config["parent_metrics_percent"])!={"U","S","H","ZS"}: raise ValueError("FVRA父指标不完整。")
    return config,sha256_file(path)


def _load_ntr_gate(config,device):
    path=Path(config["ntr_gate_model"])
    if sha256_file(path)!=config["ntr_gate_model_sha256"]: raise ValueError("FVRA父NTR gate SHA不匹配。")
    payload=torch.load(path,map_location="cpu",weights_only=False); gate=TangentStepGate(input_dim=8,max_step=1.5); gate.load_state_dict(payload["gate_state_dict"],strict=True)
    for p in gate.parameters(): p.requires_grad_(False)
    return gate.to(device).eval()


@torch.no_grad()
def evaluate(model,tensors,seenclasses,unseenclasses,device,batch_size=512):
    def pred(features,class_ids=None):
        rows=[]
        for start in range(0,features.size(0),batch_size): rows.append(model.logits(features[start:start+batch_size].to(device).float(),class_ids).argmax(dim=1).cpu())
        result=torch.cat(rows); return class_ids[result] if class_ids is not None else result
    sp=pred(tensors["seen_features"]); up=pred(tensors["unseen_features"]); zp=pred(tensors["unseen_features"],unseenclasses); s=h1.per_class_accuracy(tensors["seen_labels"],sp,seenclasses); u=h1.per_class_accuracy(tensors["unseen_labels"],up,unseenclasses); z=h1.per_class_accuracy(tensors["unseen_labels"],zp,unseenclasses); H=2*s*u/(s+u) if s+u else 0.0; return {"U":u*100,"S":s*100,"H":H*100,"ZS":z*100}


def run(config_path:Path,output_dir:Path,expected_commit:str):
    require_clean_code_tree(); code_commit=current_code_commit()
    if code_commit!=expected_commit: raise ValueError("expected-commit与当前HEAD不一致。")
    config,config_sha=load_config(config_path); base_path=Path(config["base_config"])
    if not base_path.is_absolute(): base_path=Path.cwd()/base_path
    base_config,base_config_sha=h1.load_config(base_path); paths=h1.resolve_paths(base_config); input_sha=h1.verify_inputs(base_config,paths,h1.TRAINING_KEYS); checkpoint_path=Path(config["base_checkpoint"])
    if sha256_file(checkpoint_path)!=config["base_checkpoint_sha256"]: raise ValueError("FVRA父checkpoint SHA不匹配。")
    if sha256_file(Path(config["ccgr_model"]))!=config["ccgr_model_sha256"]: raise ValueError("FVRA父CCGR model SHA不匹配。")
    device=torch.device(base_config["device"])
    if device.type!="cuda" or not torch.cuda.is_available(): raise RuntimeError("FVRA要求CUDA。")
    output_dir=prepare_output_dir(output_dir)
    with (output_dir/"config.snapshot.yaml").open("x",encoding="utf-8") as handle: yaml.safe_dump(config,handle,allow_unicode=True,sort_keys=False)
    log_handle=(output_dir/"training.log").open("x",encoding="utf-8",buffering=1); original_stdout=sys.stdout; sys.stdout=TeeStream(sys.stdout,log_handle)
    try:
        seed=int(config["seed"]); configure_reproducibility(seed,strict_determinism=True,deterministic_warn_only=False); tensors={name:torch.load(paths[name],map_location="cpu",weights_only=True) for name in ("sentence_embeds","train_features","train_labels")}; labels=tensors["train_labels"].long(); seenclasses=torch.unique(labels,sorted=True); allclasses=torch.arange(200); unseenclasses=allclasses[~torch.isin(allclasses,seenclasses)]; checkpoint=torch.load(checkpoint_path,map_location="cpu",weights_only=False); centroids=h1.visual_centroids(tensors["train_features"],labels,seenclasses); parent=VariableClassTGVPR(tensors["sentence_embeds"],seenclasses,centroids,dropout=base_config["dropout"],inner_ratio=base_config["inner_ratio"],outer_ratio=base_config["outer_ratio"],temperature=base_config["temperature"]); parent.load_state_dict(checkpoint["model_state_dict"],strict=True); parent=parent.to(device).eval(); ntr_gate=_load_ntr_gate(config,device); ntr,_=_candidate_prototypes(parent,ntr_gate,seenclasses,unseenclasses,device,"top5_vector",fixed_class_folds(seenclasses),"tangent"); base=parent.base_prototypes(); value=parent.value_candidate(allclasses.to(device)); roles=parent.semantic_group_vectors(); basis=tangent_direction_basis(base,value,roles); sim=base@base.index_select(0,seenclasses.to(device)).T; top5=sim.topk(5,dim=1).values; features=torch.stack(((base*value).sum(-1),(value-base).norm(dim=-1),text_resultant_lengths(tensors["sentence_embeds"]).to(device),top5.mean(1)),dim=1); ccgr=ClassConditionedGeometricGenerator(ntr,basis,features,unseenclasses,parent.scale(),hidden_dim=32,max_magnitude=0.2,initial_magnitude=0.02).to(device); payload=torch.load(Path(config["ccgr_model"]),map_location="cpu",weights_only=False); ccgr.load_state_dict(payload["model_state_dict"],strict=True); ccgr.eval(); prototypes=ccgr.prototypes().detach(); adapter=FeatureVisualResidualAdapter(config["hidden_dim"],config["residual_scale"]).to(device); model=FeatureAdapterClassifier(prototypes,parent.scale(),adapter).to(device); optimizer=torch.optim.Adam(adapter.parameters(),lr=float(config["lr"]),weight_decay=float(config["weight_decay"])); mapping=torch.full((200,),-1,dtype=torch.long); mapping[seenclasses]=torch.arange(150); generator=torch.Generator(device="cpu").manual_seed(seed); history=[]
        for epoch in range(1,int(config["epochs"])+1):
            permutation=torch.randperm(labels.numel(),generator=generator); loss_sum=ce_sum=consistency_sum=0.0; count=0
            for start in range(0,labels.numel(),int(config["batch_size"])):
                indices=permutation[start:start+int(config["batch_size"])]; raw=tensors["train_features"][indices].to(device).float(); targets=mapping[labels[indices]].to(device); adapted=adapter(raw); ce=F.cross_entropy(adapted@prototypes.index_select(0,seenclasses.to(device)).T*parent.scale(),targets); original=F.normalize(raw,dim=-1); consistency=1.0-(adapted*original).sum(dim=-1).mean(); loss=ce+float(config["consistency_weight"])*consistency; optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step(); loss_sum+=float(loss.detach())*indices.numel(); ce_sum+=float(ce.detach())*indices.numel(); consistency_sum+=float(consistency.detach())*indices.numel(); count+=indices.numel()
            stats=adapter.residual_stats(tensors["train_features"][:512].to(device)); row={"epoch":epoch,"loss":loss_sum/count,"ce":ce_sum/count,"consistency":consistency_sum/count,"residual":stats}; history.append(row); print(f"epoch={epoch} ce={row['ce']:.6f} consistency={row['consistency']:.6f} residual={stats}")
        torch.save({"attempt_id":config["attempt_id"],"code_commit":code_commit,"config":config,"adapter_state_dict":copy.deepcopy(adapter.state_dict()),"history":history},output_dir/"fvra_model.pth")
        # official test严格在FVRA训练结束后加载。
        input_sha.update(h1.verify_inputs(base_config,paths,h1.OFFICIAL_KEYS)); tensors.update({name:torch.load(paths[name],map_location="cpu",weights_only=True) for name in h1.OFFICIAL_KEYS}); parent_model=FeatureAdapterClassifier(prototypes,parent.scale(),FeatureVisualResidualAdapter(config["hidden_dim"],config["residual_scale"]).to(device)).to(device); parent_metrics=evaluate(parent_model,tensors,seenclasses,unseenclasses,device); candidate_metrics=evaluate(model,tensors,seenclasses,unseenclasses,device); delta={key:candidate_metrics[key]-float(config["parent_metrics_percent"][key]) for key in ("U","S","H","ZS")}; stats=adapter.residual_stats(tensors["train_features"][:512].to(device)); success=delta["H"]>=0.20 and delta["U"]>=-2 and delta["S"]>=-2 and stats["max"]<0.5; atomic_write_json(output_dir/"data_fingerprints.json",{"files":input_sha}); metrics={"attempt_id":config["attempt_id"],"idea_id":config["idea_id"],"framework_id":config["framework_id"],"code_commit":code_commit,"config_sha256":config_sha,"base_config_sha256":base_config_sha,"evaluation_protocol":h1.EVALUATION_PROTOCOL,"test_used_for_selection":True,"unseen_images_used_for_gradient":False,"recomputed_parent_metrics_percent":parent_metrics,"parent_metrics_percent":config["parent_metrics_percent"],"candidate_metrics_percent":candidate_metrics,"delta_vs_parent_percent_points":delta,"residual_stats":stats,"success":success,"fvra_model_sha256":sha256_file(output_dir/"fvra_model.pth")}; atomic_write_json(output_dir/"metrics.json",metrics); print(metrics); return metrics
    finally:
        sys.stdout.flush(); sys.stdout=original_stdout; log_handle.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--config",type=Path,required=True); parser.add_argument("--output-dir",type=Path,required=True); parser.add_argument("--expected-commit",required=True); args=parser.parse_args(); run(args.config,args.output_dir,args.expected_commit)


if __name__=="__main__": main()
