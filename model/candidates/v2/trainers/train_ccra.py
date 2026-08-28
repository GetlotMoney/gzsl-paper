from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import scipy.io as sio
import torch
import torch.nn.functional as F
import yaml

from model.candidates.v2.modules.ara import AttributeResidualAlignment,fit_ridge_attribute_map
from model.frameworks.v4.ccgr import ClassConditionedGeometricGenerator
from model.candidates.v2.modules.ccra import ClassConditionedAttributeResidual,attribute_pca_features
from model.frameworks.v2 import train as h1
from tools.reproducibility import configure_reproducibility
from tools.run_contract import atomic_write_json,current_code_commit,prepare_output_dir,require_clean_code_tree
from tools.runtime import sha256_file


CONFIG_KEYS={"schema_version","attempt_id","idea_id","framework_id","base_config","ccgr_model","ccgr_model_sha256","cra_model","cra_model_sha256","seed","epochs","batch_size","lr","weight_decay","ridge","attribute_rank","max_beta","max_beta_residual","parent_metrics_percent"}


class TeeStream:
    def __init__(self,*streams): self.streams=streams
    def write(self,value):
        for stream in self.streams: stream.write(value)
        return len(value)
    def flush(self):
        for stream in self.streams: stream.flush()


def load_config(path):
    path=Path(path).resolve(); config=yaml.safe_load(path.read_text(encoding="utf-8")); actual=set(config) if isinstance(config,dict) else set()
    if not isinstance(config,dict) or actual!=CONFIG_KEYS: raise ValueError(f"CCRA配置字段错误；缺少={sorted(CONFIG_KEYS-actual)}，多出={sorted(actual-CONFIG_KEYS)}。")
    if config["schema_version"]!="gzsl-paper.ccra.v1" or config["attempt_id"]!="V2-TRY-108" or config["idea_id"]!="IDEA-034": raise ValueError("CCRA首次TRY身份错误。")
    if int(config["epochs"])!=20 or int(config["batch_size"])!=256 or float(config["lr"])!=0.001 or float(config["weight_decay"])!=0.0001 or float(config["ridge"])!=0.01 or int(config["attribute_rank"])!=16 or float(config["max_beta"])!=20.0 or float(config["max_beta_residual"])!=4.0: raise ValueError("CCRA训练参数错误。")
    return config,sha256_file(path)


@torch.no_grad()
def evaluate(prototypes,scale,model,tensors,seenclasses,unseenclasses,device):
    def predict(features,class_ids=None):
        logits=model.logits(features.to(device).float(),prototypes,scale,class_ids); result=logits.argmax(dim=1).cpu(); return result if class_ids is None else class_ids[result]
    seen_pred=predict(tensors["seen_features"]); unseen_pred=predict(tensors["unseen_features"]); zsl_pred=predict(tensors["unseen_features"],unseenclasses); seen=h1.per_class_accuracy(tensors["seen_labels"],seen_pred,seenclasses); unseen=h1.per_class_accuracy(tensors["unseen_labels"],unseen_pred,unseenclasses); zsl=h1.per_class_accuracy(tensors["unseen_labels"],zsl_pred,unseenclasses); harmonic=2*seen*unseen/(seen+unseen) if seen+unseen else 0.0; return {"U":unseen*100,"S":seen*100,"H":harmonic*100,"ZS":zsl*100}


def run(config_path,output_dir,expected_commit):
    require_clean_code_tree(); code_commit=current_code_commit()
    if code_commit!=expected_commit: raise ValueError("expected-commit与当前HEAD不一致。")
    config,config_sha=load_config(config_path); base_path=Path(config["base_config"])
    if not base_path.is_absolute(): base_path=Path.cwd()/base_path
    base_config,base_config_sha=h1.load_config(base_path); paths=h1.resolve_paths(base_config); input_sha=h1.verify_inputs(base_config,paths,h1.TRAINING_KEYS); ccgr_path=Path(config["ccgr_model"]); cra_path=Path(config["cra_model"])
    if sha256_file(ccgr_path)!=config["ccgr_model_sha256"] or sha256_file(cra_path)!=config["cra_model_sha256"]: raise ValueError("CCRA父模型SHA不匹配。")
    attribute_sha=sha256_file(paths["att_splits"])
    if attribute_sha!=base_config["expected_sha256"]["att_splits"]: raise ValueError("CCRA属性文件SHA不匹配。")
    input_sha["att_splits"]=attribute_sha; device=torch.device(base_config["device"])
    if device.type!="cuda" or not torch.cuda.is_available(): raise RuntimeError("CCRA要求CUDA。")
    output_dir=prepare_output_dir(Path(output_dir))
    with (output_dir/"config.snapshot.yaml").open("x",encoding="utf-8") as handle: yaml.safe_dump(config,handle,allow_unicode=True,sort_keys=False)
    log_handle=(output_dir/"training.log").open("x",encoding="utf-8",buffering=1); original_stdout=sys.stdout; sys.stdout=TeeStream(sys.stdout,log_handle)
    try:
        seed=int(config["seed"]); configure_reproducibility(seed,strict_determinism=True,deterministic_warn_only=False); tensors={name:torch.load(paths[name],map_location="cpu",weights_only=True) for name in ("train_features","train_labels")}; labels=tensors["train_labels"].long(); seenclasses=torch.unique(labels,sorted=True); allclasses=torch.arange(200); unseenclasses=allclasses[~torch.isin(allclasses,seenclasses)]; ccgr_payload=torch.load(ccgr_path,map_location="cpu",weights_only=False); state=ccgr_payload["model_state_dict"]; ccgr=ClassConditionedGeometricGenerator(state["parent_prototypes"],state["direction_basis"],state["class_features"],state["target_classes"],state["_scale"],hidden_dim=state["trunk.0.weight"].shape[0],max_magnitude=ccgr_payload["config"]["max_magnitude"],initial_magnitude=ccgr_payload["config"]["initial_magnitude"]); ccgr.load_state_dict(state,strict=True); ccgr=ccgr.to(device).eval(); prototypes=ccgr.prototypes().detach(); attributes=torch.from_numpy(sio.loadmat(paths["att_splits"])["att"].T).float().to(device); centroids=h1.visual_centroids(tensors["train_features"],labels,seenclasses).to(device); ridge_weight=fit_ridge_attribute_map(centroids,seenclasses.to(device),attributes,float(config["ridge"])); base_cra=AttributeResidualAlignment(ridge_weight,attributes,config["max_beta"]); base_cra.load_state_dict(torch.load(cra_path,map_location="cpu",weights_only=False)["ara_state_dict"],strict=True); model=ClassConditionedAttributeResidual(base_cra,attribute_pca_features(attributes,config["attribute_rank"]),config["max_beta_residual"]).to(device); optimizer=torch.optim.Adam((p for p in model.parameters() if p.requires_grad),lr=float(config["lr"]),weight_decay=float(config["weight_decay"])); mapping=torch.full((200,),-1,dtype=torch.long); mapping[seenclasses]=torch.arange(seenclasses.numel()); generator=torch.Generator(device="cpu").manual_seed(seed*39000); history=[]; input_sha.update(h1.verify_inputs(base_config,paths,h1.OFFICIAL_KEYS)); tensors.update({name:torch.load(paths[name],map_location="cpu",weights_only=True) for name in h1.OFFICIAL_KEYS}); initial=evaluate(prototypes,ccgr.scale(),model,tensors,seenclasses,unseenclasses,device); best_H=initial["H"]; best_epoch=0; best_state=copy.deepcopy(model.state_dict()); history.append({"epoch":0,"official_metrics_percent":initial}); print(f"epoch=0 official_H={best_H:.6f}")
        for epoch in range(1,int(config["epochs"])+1):
            order=torch.randperm(labels.numel(),generator=generator); loss_sum=0.0; count=0
            for start in range(0,labels.numel(),int(config["batch_size"])):
                indices=order[start:start+int(config["batch_size"])]; images=tensors["train_features"][indices].to(device).float(); targets=mapping[labels[indices]].to(device); logits=model.logits(images,prototypes,ccgr.scale(),seenclasses); loss=F.cross_entropy(logits,targets); optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step(); loss_sum+=float(loss.detach()); count+=1
            official=evaluate(prototypes,ccgr.scale(),model,tensors,seenclasses,unseenclasses,device); stats=model.residual_stats(); row={"epoch":epoch,"loss":loss_sum/count,"official_metrics_percent":official,"residual_stats":stats}; history.append(row); print(f"epoch={epoch} loss={row['loss']:.6f} official_H={official['H']:.6f} residual_std={stats['std']:.6f}")
            if official["H"]>best_H: best_H=official["H"]; best_epoch=epoch; best_state=copy.deepcopy(model.state_dict())
        model.load_state_dict(best_state,strict=True); torch.save({"attempt_id":config["attempt_id"],"code_commit":code_commit,"config":config,"selected_epoch":best_epoch,"ccra_state_dict":best_state,"history":history},output_dir/"ccra_model.pth"); parent_model=ClassConditionedAttributeResidual(base_cra,attribute_pca_features(attributes,config["attribute_rank"]),config["max_beta_residual"]).to(device); parent_metrics=evaluate(prototypes,ccgr.scale(),parent_model,tensors,seenclasses,unseenclasses,device); candidate_metrics=evaluate(prototypes,ccgr.scale(),model,tensors,seenclasses,unseenclasses,device); delta={key:candidate_metrics[key]-float(config["parent_metrics_percent"][key]) for key in ("U","S","H","ZS")}; stats=model.residual_stats(); success=candidate_metrics["H"]>79.44820966336283 and delta["U"]>=-2 and delta["S"]>=-2 and stats["std"]>0.01; atomic_write_json(output_dir/"data_fingerprints.json",{"files":input_sha}); metrics={"attempt_id":config["attempt_id"],"idea_id":config["idea_id"],"framework_id":config["framework_id"],"code_commit":code_commit,"config_sha256":config_sha,"base_config_sha256":base_config_sha,"evaluation_protocol":h1.EVALUATION_PROTOCOL,"test_used_for_selection":True,"unseen_images_used_for_gradient":False,"selected_epoch":best_epoch,"attribute_rank":int(config["attribute_rank"]),"residual_stats":stats,"recomputed_parent_metrics_percent":parent_metrics,"parent_metrics_percent":config["parent_metrics_percent"],"candidate_metrics_percent":candidate_metrics,"delta_vs_parent_percent_points":delta,"success":success,"ccra_model_sha256":sha256_file(output_dir/"ccra_model.pth")}; atomic_write_json(output_dir/"metrics.json",metrics); print(metrics); return metrics
    finally:
        sys.stdout.flush(); sys.stdout=original_stdout; log_handle.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--config",type=Path,required=True); parser.add_argument("--output-dir",type=Path,required=True); parser.add_argument("--expected-commit",required=True); args=parser.parse_args(); run(args.config,args.output_dir,args.expected_commit)


if __name__=="__main__": main()
