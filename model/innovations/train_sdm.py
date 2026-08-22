from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

from model.innovations.ccgr import ClassConditionedGeometricGenerator,tangent_direction_basis
from model.innovations.dpt import text_resultant_lengths
from model.innovations.elpt import VariableClassTGVPR,fixed_class_folds
from model.innovations.sdm import SymmetricDiagonalMetric,SymmetricLowRankMetric
from model.innovations.train_elpt import _candidate_prototypes,_fold_package,_load_fold_checkpoint
from model.innovations.tst import TangentStepGate,tangent_transport
from model.tg_vpr_h1 import train as h1
from tools.reproducibility import configure_reproducibility
from tools.run_contract import atomic_write_json,current_code_commit,prepare_output_dir,require_clean_code_tree
from tools.runtime import sha256_file


CONFIG_KEYS={"schema_version","attempt_id","idea_id","framework_id","base_config","base_checkpoint","base_checkpoint_sha256","ntr_gate_model","ntr_gate_model_sha256","ccgr_model","ccgr_model_sha256","fold_checkpoint_dir","seed","epochs","batch_half","lr","weight_decay","max_log_weight","parent_metrics_percent"}
CONFIG_KEYS_LOW_RANK=CONFIG_KEYS|{"parent_sdm_model","parent_sdm_model_sha256","subspace_rank","max_subspace_log_weight"}
CONFIG_KEYS_LOW_RANK_V2=CONFIG_KEYS_LOW_RANK|{"train_base_metric"}


class TeeStream:
    def __init__(self,*streams): self.streams=streams
    def write(self,value):
        for stream in self.streams: stream.write(value)
        return len(value)
    def flush(self):
        for stream in self.streams: stream.flush()


def load_config(path:Path):
    path=path.resolve(); config=yaml.safe_load(path.read_text(encoding="utf-8")); actual=set(config) if isinstance(config,dict) else set()
    expected=CONFIG_KEYS_LOW_RANK_V2 if isinstance(config,dict) and config.get("schema_version")=="gzsl-paper.sdm-low-rank.v2" else (CONFIG_KEYS_LOW_RANK if isinstance(config,dict) and config.get("schema_version")=="gzsl-paper.sdm-low-rank.v1" else CONFIG_KEYS)
    if not isinstance(config,dict) or actual!=expected: raise ValueError(f"SDM配置字段错误；缺少={sorted(expected-actual)}，多出={sorted(actual-expected)}。")
    if config["schema_version"] not in ("gzsl-paper.sdm.v1","gzsl-paper.sdm-low-rank.v1","gzsl-paper.sdm-low-rank.v2") or config["attempt_id"] not in ("V2-TRY-086","V2-TRY-087","V2-TRY-088","V2-TRY-089","V2-TRY-090","V2-TRY-091") or config["idea_id"]!="IDEA-028": raise ValueError("SDM首次TRY身份错误。")
    if int(config["epochs"])!=20 or int(config["batch_half"])!=32 or float(config["lr"])!=0.001 or float(config["weight_decay"])!=0.0001 or float(config["max_log_weight"])!=0.1: raise ValueError("SDM训练参数错误。")
    if set(config["parent_metrics_percent"])!={"U","S","H","ZS"}: raise ValueError("SDM父指标不完整。")
    if config["attempt_id"]=="V2-TRY-087" and (int(config["subspace_rank"])!=64 or float(config["max_subspace_log_weight"])!=0.1 or not config["parent_sdm_model"]): raise ValueError("SDM低秩补救配置错误。")
    if config["attempt_id"]=="V2-TRY-088" and (int(config["subspace_rank"])!=64 or float(config["max_subspace_log_weight"])!=0.1 or not config["parent_sdm_model"] or config["train_base_metric"] is not True): raise ValueError("SDM联合低秩补救配置错误。")
    return config,sha256_file(path)


def principal_centroid_basis(centroids,rank):
    centered=F.normalize(centroids.float(),dim=-1); centered=centered-centered.mean(dim=0,keepdim=True); _,_,right=torch.linalg.svd(centered,full_matrices=False); return right[:rank]


def _load_gate(config,device):
    path=Path(config["ntr_gate_model"])
    if sha256_file(path)!=config["ntr_gate_model_sha256"]: raise ValueError("SDM父NTR gate SHA不匹配。")
    payload=torch.load(path,map_location="cpu",weights_only=False); gate=TangentStepGate(input_dim=8,max_step=1.5); gate.load_state_dict(payload["gate_state_dict"],strict=True)
    for parameter in gate.parameters(): parameter.requires_grad_(False)
    return gate.to(device).eval()


@torch.no_grad()
def evaluate(prototypes,scale,metric,tensors,seenclasses,unseenclasses,device):
    def predict(features,class_ids=None):
        selected=prototypes if class_ids is None else prototypes.index_select(0,class_ids.to(device)); logits=metric.logits(features.to(device).float(),selected,scale); result=logits.argmax(dim=1).cpu(); return result if class_ids is None else class_ids[result]
    seen_pred=predict(tensors["seen_features"]); unseen_pred=predict(tensors["unseen_features"]); zsl_pred=predict(tensors["unseen_features"],unseenclasses); seen=h1.per_class_accuracy(tensors["seen_labels"],seen_pred,seenclasses); unseen=h1.per_class_accuracy(tensors["unseen_labels"],unseen_pred,unseenclasses); zsl=h1.per_class_accuracy(tensors["unseen_labels"],zsl_pred,unseenclasses); harmonic=2*seen*unseen/(seen+unseen) if seen+unseen else 0.0; return {"U":unseen*100,"S":seen*100,"H":harmonic*100,"ZS":zsl*100}


def run(config_path:Path,output_dir:Path,expected_commit:str):
    require_clean_code_tree(); code_commit=current_code_commit()
    if code_commit!=expected_commit: raise ValueError("expected-commit与当前HEAD不一致。")
    config,config_sha=load_config(config_path); base_path=Path(config["base_config"])
    if not base_path.is_absolute(): base_path=Path.cwd()/base_path
    base_config,base_config_sha=h1.load_config(base_path); paths=h1.resolve_paths(base_config); input_sha=h1.verify_inputs(base_config,paths,h1.TRAINING_KEYS); checkpoint_path=Path(config["base_checkpoint"]); ccgr_path=Path(config["ccgr_model"])
    if sha256_file(checkpoint_path)!=config["base_checkpoint_sha256"] or sha256_file(ccgr_path)!=config["ccgr_model_sha256"]: raise ValueError("SDM父模型SHA不匹配。")
    parent_sdm_path=Path(config["parent_sdm_model"]) if config.get("parent_sdm_model") else None
    if parent_sdm_path is not None and sha256_file(parent_sdm_path)!=config["parent_sdm_model_sha256"]: raise ValueError("SDM父度量SHA不匹配。")
    device=torch.device(base_config["device"])
    if device.type!="cuda" or not torch.cuda.is_available(): raise RuntimeError("SDM要求CUDA。")
    output_dir=prepare_output_dir(output_dir)
    with (output_dir/"config.snapshot.yaml").open("x",encoding="utf-8") as handle: yaml.safe_dump(config,handle,allow_unicode=True,sort_keys=False)
    log_handle=(output_dir/"training.log").open("x",encoding="utf-8",buffering=1); original_stdout=sys.stdout; sys.stdout=TeeStream(sys.stdout,log_handle)
    try:
        seed=int(config["seed"]); configure_reproducibility(seed,strict_determinism=True,deterministic_warn_only=False); tensors={name:torch.load(paths[name],map_location="cpu",weights_only=True) for name in ("sentence_embeds","train_features","train_labels")}; labels=tensors["train_labels"].long(); seenclasses=torch.unique(labels,sorted=True); allclasses=torch.arange(200); unseenclasses=allclasses[~torch.isin(allclasses,seenclasses)]; checkpoint=torch.load(checkpoint_path,map_location="cpu",weights_only=False); centroids=h1.visual_centroids(tensors["train_features"],labels,seenclasses); parent=VariableClassTGVPR(tensors["sentence_embeds"],seenclasses,centroids,dropout=base_config["dropout"],inner_ratio=base_config["inner_ratio"],outer_ratio=base_config["outer_ratio"],temperature=base_config["temperature"]); parent.load_state_dict(checkpoint["model_state_dict"],strict=True); parent=parent.to(device).eval(); gate=_load_gate(config,device); folds=fixed_class_folds(seenclasses); ntr,_=_candidate_prototypes(parent,gate,seenclasses,unseenclasses,device,"top5_vector",folds,"tangent"); base=parent.base_prototypes(); value=parent.value_candidate(allclasses.to(device)); roles=parent.semantic_group_vectors(); basis=tangent_direction_basis(base,value,roles); top5=(base@base.index_select(0,seenclasses.to(device)).T).topk(5,dim=1).values; text_resultant=text_resultant_lengths(tensors["sentence_embeds"]).to(device); features=torch.stack(((base*value).sum(-1),(value-base).norm(dim=-1),text_resultant,top5.mean(1)),dim=1); ccgr=ClassConditionedGeometricGenerator(ntr,basis,features,unseenclasses,parent.scale(),hidden_dim=32,max_magnitude=0.2,initial_magnitude=0.02).to(device); ccgr.load_state_dict(torch.load(ccgr_path,map_location="cpu",weights_only=False)["model_state_dict"],strict=True); ccgr.eval(); final_prototypes=ccgr.prototypes().detach(); packages=[]
        for fold_id,(ps,pu) in enumerate(folds):
            fold_model=_load_fold_checkpoint(fold_id,ps,tensors["sentence_embeds"],tensors["train_features"],labels,base_config,device,config["fold_checkpoint_dir"]); package=_fold_package(fold_model,ps,pu,tensors,seenclasses,device,"top5_vector")
            with torch.no_grad():
                fb=package["base_all"].to(device); ff=package["fold_full"].to(device).clone(); fva=fold_model.value_candidate(allclasses.to(device)); fbas=tangent_direction_basis(fb,fva,fold_model.semantic_group_vectors()); ft5=(fb@fb.index_select(0,ps.to(device)).T).topk(5,dim=1).values; ffeat=torch.stack(((fb*fva).sum(-1),(fva-fb).norm(dim=-1),text_resultant,ft5.mean(1)),dim=1); pu_device=pu.to(device); ff[pu_device]=tangent_transport(fb.index_select(0,pu_device),package["value"].to(device),gate(package["gate_features"].to(device))); generated=ccgr.generate_external(ff,fbas,ffeat); ff[pu_device]=generated.index_select(0,pu_device)
            package["competition"]=ff.index_select(0,seenclasses.to(device)); packages.append(package); del fold_model
        base_metric=SymmetricDiagonalMetric(max_log_weight=config["max_log_weight"]); parent_sdm_payload=torch.load(parent_sdm_path,map_location="cpu",weights_only=False) if parent_sdm_path is not None else None
        if parent_sdm_payload is not None: base_metric.load_state_dict(parent_sdm_payload["metric_state_dict"],strict=True)
        metric=SymmetricLowRankMetric(base_metric,principal_centroid_basis(centroids,int(config["subspace_rank"])),config["max_subspace_log_weight"],freeze_base_metric=not bool(config.get("train_base_metric",False))).to(device) if parent_sdm_payload is not None else base_metric.to(device); optimizer=torch.optim.Adam((parameter for parameter in metric.parameters() if parameter.requires_grad),lr=float(config["lr"]),weight_decay=float(config["weight_decay"])); mapping=torch.full((200,),-1,dtype=torch.long); mapping[seenclasses]=torch.arange(150); generators=[torch.Generator(device="cpu").manual_seed(seed*29000+i) for i in range(3)]; half=int(config["batch_half"]); history=[]; input_sha.update(h1.verify_inputs(base_config,paths,h1.OFFICIAL_KEYS)); tensors.update({name:torch.load(paths[name],map_location="cpu",weights_only=True) for name in h1.OFFICIAL_KEYS}); initial=evaluate(final_prototypes,parent.scale(),metric,tensors,seenclasses,unseenclasses,device); best_H=initial["H"]; best_epoch=0; best_state=copy.deepcopy(metric.state_dict()); history.append({"epoch":0,"official_metrics_percent":initial,"weight_stats":metric.stats()}); print(f"epoch=0 official_H={best_H:.6f}")
        for epoch in range(1,int(config["epochs"])+1):
            loss_sum=0.0; count=0
            for fold_id,package in enumerate(packages):
                steps=min(package["seen_indices"].numel()//half,package["unseen_indices"].numel()//half)
                for _ in range(steps):
                    generator=generators[fold_id]; si=package["seen_indices"][torch.randperm(package["seen_indices"].numel(),generator=generator)[:half]]; ui=package["unseen_indices"][torch.randperm(package["unseen_indices"].numel(),generator=generator)[:half]]; indices=torch.cat((si,ui)); images=tensors["train_features"][indices].to(device).float(); targets=mapping[labels[indices]].to(device); logits=metric.logits(images,package["competition"],package["scale"].to(device)); loss=F.cross_entropy(logits,targets); optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step(); loss_sum+=float(loss.detach()); count+=1
            official=evaluate(final_prototypes,parent.scale(),metric,tensors,seenclasses,unseenclasses,device); row={"epoch":epoch,"loss":loss_sum/count,"official_metrics_percent":official,"weight_stats":metric.stats()}; history.append(row); print(f"epoch={epoch} loss={row['loss']:.6f} official_H={official['H']:.6f} weight_std={row['weight_stats']['std']:.6f}")
            if official["H"]>best_H: best_H=official["H"]; best_epoch=epoch; best_state=copy.deepcopy(metric.state_dict())
        metric.load_state_dict(best_state,strict=True); torch.save({"attempt_id":config["attempt_id"],"code_commit":code_commit,"config":config,"selected_epoch":best_epoch,"metric_state_dict":best_state,"history":history},output_dir/"sdm_model.pth"); parent_metric=SymmetricDiagonalMetric(max_log_weight=config["max_log_weight"]).to(device); parent_metric.load_state_dict(parent_sdm_payload["metric_state_dict"],strict=True) if parent_sdm_payload is not None else None; parent_metrics=evaluate(final_prototypes,parent.scale(),parent_metric,tensors,seenclasses,unseenclasses,device); candidate_metrics=evaluate(final_prototypes,parent.scale(),metric,tensors,seenclasses,unseenclasses,device); delta={key:candidate_metrics[key]-float(config["parent_metrics_percent"][key]) for key in ("U","S","H","ZS")}; stats=metric.stats(); threshold=float(config["parent_metrics_percent"]["H"]); success=candidate_metrics["H"]>threshold and delta["U"]>=-2 and delta["S"]>=-2; atomic_write_json(output_dir/"data_fingerprints.json",{"files":input_sha}); metrics={"attempt_id":config["attempt_id"],"idea_id":config["idea_id"],"framework_id":config["framework_id"],"code_commit":code_commit,"config_sha256":config_sha,"base_config_sha256":base_config_sha,"evaluation_protocol":h1.EVALUATION_PROTOCOL,"test_used_for_selection":True,"unseen_images_used_for_gradient":False,"selected_epoch":best_epoch,"recomputed_parent_metrics_percent":parent_metrics,"parent_metrics_percent":config["parent_metrics_percent"],"candidate_metrics_percent":candidate_metrics,"delta_vs_parent_percent_points":delta,"weight_stats":stats,"success":success,"sdm_model_sha256":sha256_file(output_dir/"sdm_model.pth")}; atomic_write_json(output_dir/"metrics.json",metrics); print(metrics); return metrics
    finally:
        sys.stdout.flush(); sys.stdout=original_stdout; log_handle.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--config",type=Path,required=True); parser.add_argument("--output-dir",type=Path,required=True); parser.add_argument("--expected-commit",required=True); args=parser.parse_args(); run(args.config,args.output_dir,args.expected_commit)


if __name__=="__main__": main()
