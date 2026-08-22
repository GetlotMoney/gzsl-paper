from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

from model.innovations.elpt import VariableClassTGVPR, fixed_class_folds
from model.innovations.mpr import MultiRolePrototypeClassifier
from model.innovations.train_elpt import FrozenPrototypeClassifier, _candidate_prototypes
from model.innovations.tst import TangentStepGate
from model.tg_vpr_h1 import train as h1
from tools.reproducibility import configure_reproducibility
from tools.run_contract import atomic_write_json, current_code_commit, prepare_output_dir, require_clean_code_tree
from tools.runtime import sha256_file


CONFIG_KEYS = {"schema_version","attempt_id","idea_id","framework_id","base_config","base_checkpoint","base_checkpoint_sha256","tst_gate_model","tst_gate_model_sha256","seed","epochs","batch_size","lr","weight_decay","role_temperature","max_strength","initial_strength","parent_metrics_percent"}


class TeeStream:
    def __init__(self,*streams): self.streams=streams
    def write(self,value):
        for stream in self.streams: stream.write(value)
        return len(value)
    def flush(self):
        for stream in self.streams: stream.flush()


def load_config(path: Path):
    path=path.resolve(); config=yaml.safe_load(path.read_text(encoding="utf-8")); actual=set(config) if isinstance(config,dict) else set()
    if not isinstance(config,dict) or actual!=CONFIG_KEYS: raise ValueError(f"MPR配置字段错误；缺少={sorted(CONFIG_KEYS-actual)}，多出={sorted(actual-CONFIG_KEYS)}。")
    if config["schema_version"]!="gzsl-paper.mpr.v1" or config["attempt_id"]!="V2-TRY-046" or config["idea_id"]!="IDEA-014": raise ValueError("MPR首次TRY身份错误。")
    if int(config["epochs"])!=10 or int(config["batch_size"])!=64 or float(config["lr"])!=0.01 or float(config["weight_decay"])!=0.0: raise ValueError("MPR训练参数错误。")
    if float(config["role_temperature"])!=0.05 or float(config["max_strength"])!=0.5 or float(config["initial_strength"])!=0.05: raise ValueError("MPR模块参数错误。")
    if set(config["parent_metrics_percent"])!={"U","S","H","ZS"}: raise ValueError("MPR父指标不完整。")
    return config,sha256_file(path)


def _load_gate(config,device):
    path=Path(config["tst_gate_model"])
    if sha256_file(path)!=config["tst_gate_model_sha256"]: raise ValueError("MPR父TST gate SHA不匹配。")
    payload=torch.load(path,map_location="cpu",weights_only=False); gate=TangentStepGate(input_dim=4,max_step=1.5); gate.load_state_dict(payload["gate_state_dict"],strict=True)
    for parameter in gate.parameters(): parameter.requires_grad_(False)
    return gate.to(device).eval()


@torch.no_grad()
def evaluate_mpr(model,tensors,seenclasses,unseenclasses,device,batch_size=512):
    def predict(features,class_ids=None):
        rows=[]
        for start in range(0,features.size(0),batch_size): rows.append(model.logits(features[start:start+batch_size].to(device).float(),class_ids).argmax(dim=1).cpu())
        pred=torch.cat(rows); return class_ids[pred] if class_ids is not None else pred
    seen_pred=predict(tensors["seen_features"]); unseen_pred=predict(tensors["unseen_features"]); zsl_pred=predict(tensors["unseen_features"],unseenclasses)
    seen=h1.per_class_accuracy(tensors["seen_labels"],seen_pred,seenclasses); unseen=h1.per_class_accuracy(tensors["unseen_labels"],unseen_pred,unseenclasses); zsl=h1.per_class_accuracy(tensors["unseen_labels"],zsl_pred,unseenclasses); harmonic=2*seen*unseen/(seen+unseen) if seen+unseen else 0.0
    return {"U":unseen*100,"S":seen*100,"H":harmonic*100,"ZS":zsl*100}


def run(config_path:Path,output_dir:Path,expected_commit:str):
    require_clean_code_tree(); code_commit=current_code_commit()
    if code_commit!=expected_commit: raise ValueError("expected-commit与当前HEAD不一致。")
    config,config_sha=load_config(config_path); base_path=Path(config["base_config"])
    if not base_path.is_absolute(): base_path=Path.cwd()/base_path
    base_config,base_config_sha=h1.load_config(base_path); paths=h1.resolve_paths(base_config); input_sha=h1.verify_inputs(base_config,paths,h1.TRAINING_KEYS); checkpoint_path=Path(config["base_checkpoint"])
    if sha256_file(checkpoint_path)!=config["base_checkpoint_sha256"]: raise ValueError("MPR父checkpoint SHA不匹配。")
    device=torch.device(base_config["device"])
    if device.type!="cuda" or not torch.cuda.is_available(): raise RuntimeError("MPR要求CUDA。")
    output_dir=prepare_output_dir(output_dir)
    with (output_dir/"config.snapshot.yaml").open("x",encoding="utf-8") as handle: yaml.safe_dump(config,handle,allow_unicode=True,sort_keys=False)
    log_handle=(output_dir/"training.log").open("x",encoding="utf-8",buffering=1); original_stdout=sys.stdout; sys.stdout=TeeStream(sys.stdout,log_handle)
    try:
        seed=int(config["seed"]); configure_reproducibility(seed,strict_determinism=True,deterministic_warn_only=False); tensors={name:torch.load(paths[name],map_location="cpu",weights_only=True) for name in ("sentence_embeds","train_features","train_labels")}; labels=tensors["train_labels"].long(); seenclasses=torch.unique(labels,sorted=True); allclasses=torch.arange(200); unseenclasses=allclasses[~torch.isin(allclasses,seenclasses)]; checkpoint=torch.load(checkpoint_path,map_location="cpu",weights_only=False); centroids=h1.visual_centroids(tensors["train_features"],labels,seenclasses); parent=VariableClassTGVPR(tensors["sentence_embeds"],seenclasses,centroids,dropout=base_config["dropout"],inner_ratio=base_config["inner_ratio"],outer_ratio=base_config["outer_ratio"],temperature=base_config["temperature"]); parent.load_state_dict(checkpoint["model_state_dict"],strict=True); parent=parent.to(device).eval(); gate=_load_gate(config,device); tst_prototypes,_=_candidate_prototypes(parent,gate,seenclasses,unseenclasses,device,"summary",fixed_class_folds(seenclasses),"tangent"); model=MultiRolePrototypeClassifier(tst_prototypes,parent.semantic_group_vectors(),parent.scale(),role_temperature=config["role_temperature"],max_strength=config["max_strength"],initial_strength=config["initial_strength"]).to(device); optimizer=torch.optim.Adam(model.parameters(),lr=float(config["lr"]),weight_decay=0.0); mapping=torch.full((200,),-1,dtype=torch.long); mapping[seenclasses]=torch.arange(150); generator=torch.Generator(device="cpu").manual_seed(seed); history=[]
        for epoch in range(1,int(config["epochs"])+1):
            permutation=torch.randperm(labels.numel(),generator=generator); loss_sum=0.0; count=0
            for start in range(0,labels.numel(),int(config["batch_size"])):
                indices=permutation[start:start+int(config["batch_size"])]; images=tensors["train_features"][indices].to(device).float(); targets=mapping[labels[indices]].to(device); loss=F.cross_entropy(model.logits(images,seenclasses),targets); optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step(); loss_sum+=float(loss.detach())*indices.numel(); count+=indices.numel()
            row={"epoch":epoch,"ce":loss_sum/count,"strength":float(model.strength().detach())}; history.append(row); print(f"epoch={epoch} ce={row['ce']:.6f} strength={row['strength']:.6f}")
        torch.save({"attempt_id":config["attempt_id"],"code_commit":code_commit,"config":config,"model_state_dict":copy.deepcopy(model.state_dict()),"history":history},output_dir/"mpr_model.pth")
        # official test严格在MPR训练结束后加载。
        input_sha.update(h1.verify_inputs(base_config,paths,h1.OFFICIAL_KEYS)); tensors.update({name:torch.load(paths[name],map_location="cpu",weights_only=True) for name in h1.OFFICIAL_KEYS}); parent_metrics=h1.evaluate(FrozenPrototypeClassifier(tst_prototypes,parent.scale()).to(device),tensors,seenclasses,unseenclasses,device); candidate_metrics=evaluate_mpr(model,tensors,seenclasses,unseenclasses,device); delta={key:candidate_metrics[key]-float(config["parent_metrics_percent"][key]) for key in ("U","S","H","ZS")}; strength=float(model.strength().detach()); success=delta["H"]>=0.20 and delta["U"]>=-2 and delta["S"]>=-2 and strength<0.49; atomic_write_json(output_dir/"data_fingerprints.json",{"files":input_sha}); metrics={"attempt_id":config["attempt_id"],"idea_id":config["idea_id"],"framework_id":config["framework_id"],"code_commit":code_commit,"config_sha256":config_sha,"base_config_sha256":base_config_sha,"evaluation_protocol":h1.EVALUATION_PROTOCOL,"test_used_for_selection":True,"unseen_images_used_for_gradient":False,"recomputed_parent_metrics_percent":parent_metrics,"parent_metrics_percent":config["parent_metrics_percent"],"candidate_metrics_percent":candidate_metrics,"delta_vs_parent_percent_points":delta,"learned_strength":strength,"success":success,"mpr_model_sha256":sha256_file(output_dir/"mpr_model.pth")}; atomic_write_json(output_dir/"metrics.json",metrics); print(metrics); return metrics
    finally:
        sys.stdout.flush(); sys.stdout=original_stdout; log_handle.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--config",type=Path,required=True); parser.add_argument("--output-dir",type=Path,required=True); parser.add_argument("--expected-commit",required=True); args=parser.parse_args(); run(args.config,args.output_dir,args.expected_commit)


if __name__=="__main__": main()
