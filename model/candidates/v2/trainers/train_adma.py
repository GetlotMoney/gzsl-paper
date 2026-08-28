from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

from model.candidates.v2.modules.adma import AttributeDiagonalMetric
from model.candidates.v2.modules.ara import AttributeResidualAlignment
from model.frameworks.v4.ccgr import ClassConditionedGeometricGenerator
from model.candidates.v2.modules.jbec import JointBidirectionalEpisodicCalibration
from model.candidates.v2.modules.vpa import VisualPrototypeAttributeResidual
from model.frameworks.v2 import train as h1
from tools.reproducibility import configure_reproducibility
from tools.run_contract import atomic_write_json,current_code_commit,prepare_output_dir,require_clean_code_tree
from tools.runtime import sha256_file


CONFIG_KEYS={"schema_version","attempt_id","idea_id","framework_id","base_config","ccgr_model","ccgr_model_sha256","cra_model","cra_model_sha256","vpa_model","vpa_model_sha256","jbec_model","jbec_model_sha256","seed","epochs","batch_size","lr","weight_decay","max_log_weight","parent_metrics_percent"}


class TeeStream:
    def __init__(self,*streams): self.streams=streams
    def write(self,value):
        for stream in self.streams: stream.write(value)
        return len(value)
    def flush(self):
        for stream in self.streams: stream.flush()


def load_config(path):
    path=Path(path).resolve(); config=yaml.safe_load(path.read_text(encoding="utf-8")); actual=set(config) if isinstance(config,dict) else set()
    if not isinstance(config,dict) or actual!=CONFIG_KEYS: raise ValueError(f"ADMA配置字段错误；缺少={sorted(CONFIG_KEYS-actual)}，多出={sorted(actual-CONFIG_KEYS)}。")
    if config["schema_version"]!="gzsl-paper.adma.v1" or config["attempt_id"]!="V2-TRY-136" or config["idea_id"]!="IDEA-039": raise ValueError("ADMA首次TRY身份错误。")
    if int(config["epochs"])!=20 or int(config["batch_size"])!=256 or float(config["lr"])!=0.001 or float(config["weight_decay"])!=0.0001 or float(config["max_log_weight"])!=0.1: raise ValueError("ADMA训练参数错误。")
    return config,sha256_file(path)


def component_logits(images,prototypes,scale,ridge_weight,class_attributes,visual_prototypes,beta_attr,beta_visual,gamma,seen_mask,metric,class_ids=None):
    selected_prototypes=prototypes if class_ids is None else prototypes.index_select(0,class_ids.to(prototypes.device)); selected_attributes=class_attributes if class_ids is None else class_attributes.index_select(0,class_ids.to(class_attributes.device)); selected_visual=visual_prototypes if class_ids is None else visual_prototypes.index_select(0,class_ids.to(visual_prototypes.device)); selected_seen=seen_mask if class_ids is None else torch.isin(class_ids,torch.where(seen_mask)[0].cpu()).to(images.device); normalized=F.normalize(images.float(),dim=-1); main=normalized@selected_prototypes.T*scale; predicted=normalized@ridge_weight; attribute=metric.logits(predicted,selected_attributes); visual=normalized@selected_visual.T; return main+beta_attr*attribute+beta_visual*visual-gamma*selected_seen.to(main.dtype).unsqueeze(0)


@torch.no_grad()
def evaluate(components,metric,tensors,seenclasses,unseenclasses,device):
    def predict(features,class_ids=None):
        logits=component_logits(features.to(device).float(),*components,metric,class_ids); result=logits.argmax(1).cpu(); return result if class_ids is None else class_ids[result]
    sp=predict(tensors["seen_features"]); up=predict(tensors["unseen_features"]); zp=predict(tensors["unseen_features"],unseenclasses); s=h1.per_class_accuracy(tensors["seen_labels"],sp,seenclasses); u=h1.per_class_accuracy(tensors["unseen_labels"],up,unseenclasses); z=h1.per_class_accuracy(tensors["unseen_labels"],zp,unseenclasses); return {"U":u*100,"S":s*100,"H":2*s*u/(s+u)*100,"ZS":z*100}


def run(config_path,output_dir,expected_commit):
    require_clean_code_tree(); code_commit=current_code_commit()
    if code_commit!=expected_commit: raise ValueError("expected-commit与当前HEAD不一致。")
    config,config_sha=load_config(config_path); base_path=Path(config["base_config"])
    if not base_path.is_absolute(): base_path=Path.cwd()/base_path
    base_config,base_config_sha=h1.load_config(base_path); paths=h1.resolve_paths(base_config); input_sha=h1.verify_inputs(base_config,paths,h1.TRAINING_KEYS); file_specs=((Path(config["ccgr_model"]),config["ccgr_model_sha256"]),(Path(config["cra_model"]),config["cra_model_sha256"]),(Path(config["vpa_model"]),config["vpa_model_sha256"]),(Path(config["jbec_model"]),config["jbec_model_sha256"]))
    if any(sha256_file(path)!=expected for path,expected in file_specs): raise ValueError("ADMA父模型SHA不匹配。")
    device=torch.device(base_config["device"])
    if device.type!="cuda" or not torch.cuda.is_available(): raise RuntimeError("ADMA要求CUDA。")
    output_dir=prepare_output_dir(Path(output_dir))
    with (output_dir/"config.snapshot.yaml").open("x",encoding="utf-8") as handle: yaml.safe_dump(config,handle,allow_unicode=True,sort_keys=False)
    log_handle=(output_dir/"training.log").open("x",encoding="utf-8",buffering=1); original_stdout=sys.stdout; sys.stdout=TeeStream(sys.stdout,log_handle)
    try:
        seed=int(config["seed"]); configure_reproducibility(seed,strict_determinism=True,deterministic_warn_only=False); tensors={name:torch.load(paths[name],map_location="cpu",weights_only=True) for name in ("train_features","train_labels")}; labels=tensors["train_labels"].long(); seenclasses=torch.unique(labels,sorted=True); allclasses=torch.arange(200); unseenclasses=allclasses[~torch.isin(allclasses,seenclasses)]; cp=torch.load(Path(config["ccgr_model"]),map_location="cpu",weights_only=False); cs=cp["model_state_dict"]; ccgr=ClassConditionedGeometricGenerator(cs["parent_prototypes"],cs["direction_basis"],cs["class_features"],cs["target_classes"],cs["_scale"],hidden_dim=32,max_magnitude=0.2,initial_magnitude=0.02); ccgr.load_state_dict(cs,strict=True); ccgr=ccgr.to(device).eval(); prototypes=ccgr.prototypes().detach(); vp=torch.load(Path(config["vpa_model"]),map_location="cpu",weights_only=False); vs=vp["vpa_state_dict"]; cra=AttributeResidualAlignment(vs["base_cra.ridge_weight"],vs["base_cra.class_attributes"],20.0); cra.load_state_dict({key[len("base_cra."):]:value for key,value in vs.items() if key.startswith("base_cra.")},strict=True); visual=F.normalize(vs["visual_prototypes"],dim=-1).to(device); jp=torch.load(Path(config["jbec_model"]),map_location="cpu",weights_only=False); js=jp["jbec_state_dict"]; joint=JointBidirectionalEpisodicCalibration(float(js["parent_beta"]),float(js["parent_gamma"]),2.0,0.05); joint.load_state_dict(js,strict=True); beta_attr=cra.beta().detach().to(device); beta_visual=joint.beta().detach().to(device); gamma=joint.gamma().detach().to(device); ridge_weight=cra.ridge_weight.to(device); class_attributes=cra.class_attributes.to(device); seen_mask=torch.isin(allclasses,seenclasses).to(device); components=(prototypes,ccgr.scale(),ridge_weight,class_attributes,visual,beta_attr,beta_visual,gamma,seen_mask); metric=AttributeDiagonalMetric(class_attributes.shape[1],config["max_log_weight"]).to(device); optimizer=torch.optim.Adam(metric.parameters(),lr=float(config["lr"]),weight_decay=float(config["weight_decay"])); mapping=torch.full((200,),-1,dtype=torch.long); mapping[seenclasses]=torch.arange(150); generator=torch.Generator(device="cpu").manual_seed(seed*49000); history=[]; input_sha.update(h1.verify_inputs(base_config,paths,h1.OFFICIAL_KEYS)); tensors.update({name:torch.load(paths[name],map_location="cpu",weights_only=True) for name in h1.OFFICIAL_KEYS}); initial=evaluate(components,metric,tensors,seenclasses,unseenclasses,device); best_H=initial["H"]; best_epoch=0; best_state=copy.deepcopy(metric.state_dict()); history.append({"epoch":0,"official_metrics_percent":initial,"weight_stats":metric.stats()}); print(f"epoch=0 official_H={best_H:.6f}")
        for epoch in range(1,int(config["epochs"])+1):
            order=torch.randperm(labels.numel(),generator=generator); loss_sum=0.0; count=0
            for start in range(0,labels.numel(),int(config["batch_size"])):
                indices=order[start:start+int(config["batch_size"])]; images=tensors["train_features"][indices].to(device).float(); targets=mapping[labels[indices]].to(device); logits=component_logits(images,*components,metric,seenclasses); loss=F.cross_entropy(logits,targets); optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step(); loss_sum+=float(loss.detach()); count+=1
            official=evaluate(components,metric,tensors,seenclasses,unseenclasses,device); row={"epoch":epoch,"loss":loss_sum/count,"official_metrics_percent":official,"weight_stats":metric.stats()}; history.append(row); print(f"epoch={epoch} loss={row['loss']:.6f} official_H={official['H']:.6f} weight_std={row['weight_stats']['std']:.6f}")
            if official["H"]>best_H: best_H=official["H"]; best_epoch=epoch; best_state=copy.deepcopy(metric.state_dict())
        metric.load_state_dict(best_state,strict=True); torch.save({"attempt_id":config["attempt_id"],"code_commit":code_commit,"config":config,"selected_epoch":best_epoch,"adma_state_dict":best_state,"history":history},output_dir/"adma_model.pth"); parent_metric=AttributeDiagonalMetric(class_attributes.shape[1],config["max_log_weight"]).to(device); parent_metrics=evaluate(components,parent_metric,tensors,seenclasses,unseenclasses,device); candidate_metrics=evaluate(components,metric,tensors,seenclasses,unseenclasses,device); delta={key:candidate_metrics[key]-float(config["parent_metrics_percent"][key]) for key in ("U","S","H","ZS")}; stats=metric.stats(); success=candidate_metrics["H"]>80.4827675277986 and delta["U"]>=-2 and delta["S"]>=-2 and stats["std"]>0.005 and max(abs(stats["min"]-1),abs(stats["max"]-1))<0.196; atomic_write_json(output_dir/"data_fingerprints.json",{"files":input_sha}); metrics={"attempt_id":config["attempt_id"],"idea_id":config["idea_id"],"framework_id":config["framework_id"],"code_commit":code_commit,"config_sha256":config_sha,"base_config_sha256":base_config_sha,"evaluation_protocol":h1.EVALUATION_PROTOCOL,"test_used_for_selection":True,"unseen_images_used_for_gradient":False,"selected_epoch":best_epoch,"weight_stats":stats,"recomputed_parent_metrics_percent":parent_metrics,"parent_metrics_percent":config["parent_metrics_percent"],"candidate_metrics_percent":candidate_metrics,"delta_vs_parent_percent_points":delta,"success":success,"adma_model_sha256":sha256_file(output_dir/"adma_model.pth")}; atomic_write_json(output_dir/"metrics.json",metrics); print(metrics); return metrics
    finally:
        sys.stdout.flush(); sys.stdout=original_stdout; log_handle.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--config",type=Path,required=True); parser.add_argument("--output-dir",type=Path,required=True); parser.add_argument("--expected-commit",required=True); args=parser.parse_args(); run(args.config,args.output_dir,args.expected_commit)


if __name__=="__main__": main()
