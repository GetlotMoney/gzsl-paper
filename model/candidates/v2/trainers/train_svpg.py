from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

from model.frameworks.v4.tg import VariableClassTGVPR, fixed_class_folds, topology_loss
from model.candidates.v2.modules.svpg import SemanticVisualPrototypeGenerator
from model.candidates.v2.trainers.train_elpt import FrozenPrototypeClassifier, _candidate_prototypes
from model.frameworks.v4.tst import TangentStepGate
from model.frameworks.v2 import train as h1
from tools.reproducibility import configure_reproducibility
from tools.run_contract import atomic_write_json, current_code_commit, prepare_output_dir, require_clean_code_tree
from tools.runtime import sha256_file


CONFIG_KEYS={"schema_version","attempt_id","idea_id","framework_id","base_config","base_checkpoint","base_checkpoint_sha256","tst_gate_model","tst_gate_model_sha256","seed","epochs","batch_size","lr","weight_decay","hidden_dim","residual_scale","topology_weight","parent_metrics_percent"}
CONFIG_KEYS_V2=CONFIG_KEYS|{"training_objective","apply_scope"}
CONFIG_KEYS_V3=CONFIG_KEYS_V2|{"max_residual_norm"}


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
    expected=CONFIG_KEYS_V3 if schema=="gzsl-paper.svpg.v3" else (CONFIG_KEYS_V2 if schema=="gzsl-paper.svpg.v2" else CONFIG_KEYS)
    if not isinstance(config,dict) or actual!=expected: raise ValueError(f"SVPG配置字段错误；缺少={sorted(expected-actual)}，多出={sorted(actual-expected)}。")
    if config["schema_version"] not in ("gzsl-paper.svpg.v1","gzsl-paper.svpg.v2","gzsl-paper.svpg.v3") or config["attempt_id"] not in ("V2-TRY-053","V2-TRY-054","V2-TRY-055") or config["idea_id"]!="IDEA-016": raise ValueError("SVPG首次TRY身份错误。")
    expected_epochs=20 if config["attempt_id"]=="V2-TRY-053" else 200
    if int(config["epochs"])!=expected_epochs or int(config["batch_size"])!=64 or float(config["lr"])!=0.001 or float(config["weight_decay"])!=0.0001: raise ValueError("SVPG训练参数错误。")
    if int(config["hidden_dim"])!=128 or float(config["residual_scale"])!=0.1 or float(config["topology_weight"])!=0.1: raise ValueError("SVPG模块参数错误。")
    if set(config["parent_metrics_percent"])!={"U","S","H","ZS"}: raise ValueError("SVPG父指标不完整。")
    config.setdefault("training_objective","seen_image_ce"); config.setdefault("apply_scope","all_classes")
    config.setdefault("max_residual_norm",None)
    expected_objective="seen_image_ce" if config["attempt_id"]=="V2-TRY-053" else "seen_centroid_alignment"
    expected_scope="all_classes" if config["attempt_id"]=="V2-TRY-053" else "unseen_only"
    if config["training_objective"]!=expected_objective or config["apply_scope"]!=expected_scope: raise ValueError("SVPG训练目标或应用范围错误。")
    expected_bound=0.2 if config["attempt_id"]=="V2-TRY-055" else None
    if config["max_residual_norm"]!=expected_bound: raise ValueError("SVPG残差边界与TRY身份错误。")
    return config,sha256_file(path)


def _load_gate(config,device):
    path=Path(config["tst_gate_model"])
    if sha256_file(path)!=config["tst_gate_model_sha256"]: raise ValueError("SVPG父TST gate SHA不匹配。")
    payload=torch.load(path,map_location="cpu",weights_only=False); gate=TangentStepGate(input_dim=4,max_step=1.5); gate.load_state_dict(payload["gate_state_dict"],strict=True)
    for p in gate.parameters(): p.requires_grad_(False)
    return gate.to(device).eval()


def run(config_path:Path,output_dir:Path,expected_commit:str):
    require_clean_code_tree(); code_commit=current_code_commit()
    if code_commit!=expected_commit: raise ValueError("expected-commit与当前HEAD不一致。")
    config,config_sha=load_config(config_path); base_path=Path(config["base_config"])
    if not base_path.is_absolute(): base_path=Path.cwd()/base_path
    base_config,base_config_sha=h1.load_config(base_path); paths=h1.resolve_paths(base_config); input_sha=h1.verify_inputs(base_config,paths,h1.TRAINING_KEYS); checkpoint_path=Path(config["base_checkpoint"])
    if sha256_file(checkpoint_path)!=config["base_checkpoint_sha256"]: raise ValueError("SVPG父checkpoint SHA不匹配。")
    device=torch.device(base_config["device"])
    if device.type!="cuda" or not torch.cuda.is_available(): raise RuntimeError("SVPG要求CUDA。")
    output_dir=prepare_output_dir(output_dir)
    with (output_dir/"config.snapshot.yaml").open("x",encoding="utf-8") as handle: yaml.safe_dump(config,handle,allow_unicode=True,sort_keys=False)
    log_handle=(output_dir/"training.log").open("x",encoding="utf-8",buffering=1); original_stdout=sys.stdout; sys.stdout=TeeStream(sys.stdout,log_handle)
    try:
        seed=int(config["seed"]); configure_reproducibility(seed,strict_determinism=True,deterministic_warn_only=False); tensors={name:torch.load(paths[name],map_location="cpu",weights_only=True) for name in ("sentence_embeds","train_features","train_labels")}; labels=tensors["train_labels"].long(); seenclasses=torch.unique(labels,sorted=True); allclasses=torch.arange(200); unseenclasses=allclasses[~torch.isin(allclasses,seenclasses)]; checkpoint=torch.load(checkpoint_path,map_location="cpu",weights_only=False); centroids=h1.visual_centroids(tensors["train_features"],labels,seenclasses); parent=VariableClassTGVPR(tensors["sentence_embeds"],seenclasses,centroids,dropout=base_config["dropout"],inner_ratio=base_config["inner_ratio"],outer_ratio=base_config["outer_ratio"],temperature=base_config["temperature"]); parent.load_state_dict(checkpoint["model_state_dict"],strict=True); parent=parent.to(device).eval(); gate=_load_gate(config,device); tst_prototypes,_=_candidate_prototypes(parent,gate,seenclasses,unseenclasses,device,"summary",fixed_class_folds(seenclasses),"tangent"); target_classes=unseenclasses if config["apply_scope"]=="unseen_only" else None; model=SemanticVisualPrototypeGenerator(tst_prototypes,parent.scale(),hidden_dim=config["hidden_dim"],residual_scale=config["residual_scale"],target_classes=target_classes,max_residual_norm=config["max_residual_norm"]).to(device); optimizer=torch.optim.Adam(model.parameters(),lr=float(config["lr"]),weight_decay=float(config["weight_decay"])); mapping=torch.full((200,),-1,dtype=torch.long); mapping[seenclasses]=torch.arange(150); generator=torch.Generator(device="cpu").manual_seed(seed); history=[]
        for epoch in range(1,int(config["epochs"])+1):
            if config["training_objective"]=="seen_centroid_alignment":
                generated=model.generated_all(); generated_seen=generated.index_select(0,seenclasses.to(device)); alignment=1.0-(generated_seen*centroids.to(device)).sum(dim=-1).mean(); topo=topology_loss(tst_prototypes,generated); loss=alignment+float(config["topology_weight"])*topo; optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step(); stats=model.residual_stats(); row={"epoch":epoch,"loss":float(loss.detach()),"alignment":float(alignment.detach()),"topology":float(topo.detach()),"residual":stats}; history.append(row)
                if epoch in (1,10,20,50,100,150,200): print(f"epoch={epoch} alignment={row['alignment']:.6f} topology={row['topology']:.6f} residual={stats}")
                continue
            permutation=torch.randperm(labels.numel(),generator=generator); loss_sum=ce_sum=topo_sum=0.0; count=0
            for start in range(0,labels.numel(),int(config["batch_size"])):
                indices=permutation[start:start+int(config["batch_size"])]; images=tensors["train_features"][indices].to(device).float(); targets=mapping[labels[indices]].to(device); ce=F.cross_entropy(model.logits(images,seenclasses),targets); topo=model.topology_loss(); loss=ce+float(config["topology_weight"])*topo; optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step(); loss_sum+=float(loss.detach())*indices.numel(); ce_sum+=float(ce.detach())*indices.numel(); topo_sum+=float(topo.detach())*indices.numel(); count+=indices.numel()
            stats=model.residual_stats(); row={"epoch":epoch,"loss":loss_sum/count,"ce":ce_sum/count,"topology":topo_sum/count,"residual":stats}; history.append(row); print(f"epoch={epoch} ce={row['ce']:.6f} topology={row['topology']:.6f} residual={stats}")
        torch.save({"attempt_id":config["attempt_id"],"code_commit":code_commit,"config":config,"model_state_dict":copy.deepcopy(model.state_dict()),"history":history},output_dir/"svpg_model.pth")
        # official test严格在SVPG训练结束后加载。
        input_sha.update(h1.verify_inputs(base_config,paths,h1.OFFICIAL_KEYS)); tensors.update({name:torch.load(paths[name],map_location="cpu",weights_only=True) for name in h1.OFFICIAL_KEYS}); parent_metrics=h1.evaluate(FrozenPrototypeClassifier(tst_prototypes,parent.scale()).to(device),tensors,seenclasses,unseenclasses,device); candidate_metrics=h1.evaluate(model,tensors,seenclasses,unseenclasses,device); delta={key:candidate_metrics[key]-float(config["parent_metrics_percent"][key]) for key in ("U","S","H","ZS")}; stats=model.residual_stats(); success=delta["H"]>=0.20 and delta["U"]>=-2 and delta["S"]>=-2 and stats["max"]<0.5; atomic_write_json(output_dir/"data_fingerprints.json",{"files":input_sha}); metrics={"attempt_id":config["attempt_id"],"idea_id":config["idea_id"],"framework_id":config["framework_id"],"code_commit":code_commit,"config_sha256":config_sha,"base_config_sha256":base_config_sha,"evaluation_protocol":h1.EVALUATION_PROTOCOL,"test_used_for_selection":True,"unseen_images_used_for_gradient":False,"recomputed_parent_metrics_percent":parent_metrics,"parent_metrics_percent":config["parent_metrics_percent"],"candidate_metrics_percent":candidate_metrics,"delta_vs_parent_percent_points":delta,"training_objective":config["training_objective"],"apply_scope":config["apply_scope"],"residual_stats":stats,"success":success,"svpg_model_sha256":sha256_file(output_dir/"svpg_model.pth")}; atomic_write_json(output_dir/"metrics.json",metrics); print(metrics); return metrics
    finally:
        sys.stdout.flush(); sys.stdout=original_stdout; log_handle.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--config",type=Path,required=True); parser.add_argument("--output-dir",type=Path,required=True); parser.add_argument("--expected-commit",required=True); args=parser.parse_args(); run(args.config,args.output_dir,args.expected_commit)


if __name__=="__main__": main()
