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
from model.frameworks.v4.ccgr import ClassConditionedGeometricGenerator,tangent_direction_basis
from model.candidates.v2.modules.dpt import text_resultant_lengths
from model.candidates.v2.modules.ebc import EpisodicBiasCalibration
from model.frameworks.v4.tg import VariableClassTGVPR,fixed_class_folds
from model.candidates.v2.modules.sdm import SymmetricDiagonalMetric
from model.candidates.v2.trainers.train_elpt import _candidate_prototypes,_fold_package,_load_fold_checkpoint
from model.frameworks.v4.tst import TangentStepGate,tangent_transport
from model.frameworks.v2 import train as h1
from tools.reproducibility import configure_reproducibility
from tools.run_contract import atomic_write_json,current_code_commit,prepare_output_dir,require_clean_code_tree
from tools.runtime import sha256_file


CONFIG_KEYS={"schema_version","attempt_id","idea_id","framework_id","base_config","base_checkpoint","base_checkpoint_sha256","ntr_gate_model","ntr_gate_model_sha256","ccgr_model","ccgr_model_sha256","cra_model","cra_model_sha256","fold_checkpoint_dir","seed","epochs","batch_half","lr","weight_decay","ridge","max_beta","max_gamma","parent_metrics_percent"}


class TeeStream:
    def __init__(self,*streams): self.streams=streams
    def write(self,value):
        for stream in self.streams: stream.write(value)
        return len(value)
    def flush(self):
        for stream in self.streams: stream.flush()


def load_config(path):
    path=Path(path).resolve(); config=yaml.safe_load(path.read_text(encoding="utf-8")); actual=set(config) if isinstance(config,dict) else set()
    if not isinstance(config,dict) or actual!=CONFIG_KEYS: raise ValueError(f"EBC配置字段错误；缺少={sorted(CONFIG_KEYS-actual)}，多出={sorted(actual-CONFIG_KEYS)}。")
    if config["schema_version"]!="gzsl-paper.ebc.v1" or config["attempt_id"] not in ("V2-TRY-113","V2-TRY-114","V2-TRY-115","V2-TRY-116","V2-TRY-117") or config["idea_id"]!="IDEA-035": raise ValueError("EBC首次TRY身份错误。")
    expected_gamma=0.15 if config["attempt_id"] in ("V2-TRY-114","V2-TRY-115","V2-TRY-116","V2-TRY-117") else 0.2
    if int(config["epochs"])!=20 or int(config["batch_half"])!=32 or float(config["lr"])!=0.01 or float(config["weight_decay"])!=0.0 or float(config["ridge"])!=0.01 or float(config["max_beta"])!=20.0 or float(config["max_gamma"])!=expected_gamma: raise ValueError("EBC训练参数错误。")
    return config,sha256_file(path)


def _load_gate(config,device):
    path=Path(config["ntr_gate_model"])
    if sha256_file(path)!=config["ntr_gate_model_sha256"]: raise ValueError("EBC父NTR gate SHA不匹配。")
    payload=torch.load(path,map_location="cpu",weights_only=False); gate=TangentStepGate(input_dim=8,max_step=1.5); gate.load_state_dict(payload["gate_state_dict"],strict=True)
    for parameter in gate.parameters(): parameter.requires_grad_(False)
    return gate.to(device).eval()


@torch.no_grad()
def evaluate(prototypes,scale,cra,calibrator,tensors,seenclasses,unseenclasses,device):
    identity=SymmetricDiagonalMetric().to(device); seen_mask=torch.isin(torch.arange(200),seenclasses).to(device)
    def full_predict(features):
        logits=cra.logits(features.to(device).float(),prototypes,scale,identity); return calibrator(logits,seen_mask).argmax(dim=1).cpu()
    seen_pred=full_predict(tensors["seen_features"]); unseen_pred=full_predict(tensors["unseen_features"]); zsl_logits=cra.logits(tensors["unseen_features"].to(device).float(),prototypes,scale,identity,unseenclasses); zsl_pred=unseenclasses[zsl_logits.argmax(dim=1).cpu()]; seen=h1.per_class_accuracy(tensors["seen_labels"],seen_pred,seenclasses); unseen=h1.per_class_accuracy(tensors["unseen_labels"],unseen_pred,unseenclasses); zsl=h1.per_class_accuracy(tensors["unseen_labels"],zsl_pred,unseenclasses); harmonic=2*seen*unseen/(seen+unseen) if seen+unseen else 0.0; return {"U":unseen*100,"S":seen*100,"H":harmonic*100,"ZS":zsl*100}


def run(config_path,output_dir,expected_commit):
    require_clean_code_tree(); code_commit=current_code_commit()
    if code_commit!=expected_commit: raise ValueError("expected-commit与当前HEAD不一致。")
    config,config_sha=load_config(config_path); base_path=Path(config["base_config"])
    if not base_path.is_absolute(): base_path=Path.cwd()/base_path
    base_config,base_config_sha=h1.load_config(base_path); paths=h1.resolve_paths(base_config); input_sha=h1.verify_inputs(base_config,paths,h1.TRAINING_KEYS); checkpoint_path=Path(config["base_checkpoint"]); ccgr_path=Path(config["ccgr_model"]); cra_path=Path(config["cra_model"])
    if sha256_file(checkpoint_path)!=config["base_checkpoint_sha256"] or sha256_file(ccgr_path)!=config["ccgr_model_sha256"] or sha256_file(cra_path)!=config["cra_model_sha256"]: raise ValueError("EBC父模型SHA不匹配。")
    attribute_sha=sha256_file(paths["att_splits"])
    if attribute_sha!=base_config["expected_sha256"]["att_splits"]: raise ValueError("EBC属性文件SHA不匹配。")
    input_sha["att_splits"]=attribute_sha; device=torch.device(base_config["device"])
    if device.type!="cuda" or not torch.cuda.is_available(): raise RuntimeError("EBC要求CUDA。")
    output_dir=prepare_output_dir(Path(output_dir))
    with (output_dir/"config.snapshot.yaml").open("x",encoding="utf-8") as handle: yaml.safe_dump(config,handle,allow_unicode=True,sort_keys=False)
    log_handle=(output_dir/"training.log").open("x",encoding="utf-8",buffering=1); original_stdout=sys.stdout; sys.stdout=TeeStream(sys.stdout,log_handle)
    try:
        seed=int(config["seed"]); configure_reproducibility(seed,strict_determinism=True,deterministic_warn_only=False); tensors={name:torch.load(paths[name],map_location="cpu",weights_only=True) for name in ("sentence_embeds","train_features","train_labels")}; labels=tensors["train_labels"].long(); seenclasses=torch.unique(labels,sorted=True); allclasses=torch.arange(200); unseenclasses=allclasses[~torch.isin(allclasses,seenclasses)]; centroids=h1.visual_centroids(tensors["train_features"],labels,seenclasses); checkpoint=torch.load(checkpoint_path,map_location="cpu",weights_only=False); parent=VariableClassTGVPR(tensors["sentence_embeds"],seenclasses,centroids,dropout=base_config["dropout"],inner_ratio=base_config["inner_ratio"],outer_ratio=base_config["outer_ratio"],temperature=base_config["temperature"]); parent.load_state_dict(checkpoint["model_state_dict"],strict=True); parent=parent.to(device).eval(); gate=_load_gate(config,device); folds=fixed_class_folds(seenclasses); ntr,_=_candidate_prototypes(parent,gate,seenclasses,unseenclasses,device,"top5_vector",folds,"tangent"); base=parent.base_prototypes(); value=parent.value_candidate(allclasses.to(device)); roles=parent.semantic_group_vectors(); basis=tangent_direction_basis(base,value,roles); top5=(base@base.index_select(0,seenclasses.to(device)).T).topk(5,dim=1).values; text_resultant=text_resultant_lengths(tensors["sentence_embeds"]).to(device); features=torch.stack(((base*value).sum(-1),(value-base).norm(dim=-1),text_resultant,top5.mean(1)),dim=1); ccgr=ClassConditionedGeometricGenerator(ntr,basis,features,unseenclasses,parent.scale(),hidden_dim=32,max_magnitude=0.2,initial_magnitude=0.02).to(device); ccgr.load_state_dict(torch.load(ccgr_path,map_location="cpu",weights_only=False)["model_state_dict"],strict=True); ccgr.eval(); final_prototypes=ccgr.prototypes().detach(); attributes=torch.from_numpy(sio.loadmat(paths["att_splits"])["att"].T).float().to(device); main_ridge=fit_ridge_attribute_map(centroids.to(device),seenclasses.to(device),attributes,float(config["ridge"])); main_cra=AttributeResidualAlignment(main_ridge,attributes,config["max_beta"]); main_cra.load_state_dict(torch.load(cra_path,map_location="cpu",weights_only=False)["ara_state_dict"],strict=True); main_cra=main_cra.to(device).eval(); packages=[]
        for fold_id,(ps,pu) in enumerate(folds):
            fold_model=_load_fold_checkpoint(fold_id,ps,tensors["sentence_embeds"],tensors["train_features"],labels,base_config,device,config["fold_checkpoint_dir"]); package=_fold_package(fold_model,ps,pu,tensors,seenclasses,device,"top5_vector")
            with torch.no_grad():
                fb=package["base_all"].to(device); ff=package["fold_full"].to(device).clone(); fva=fold_model.value_candidate(allclasses.to(device)); fbas=tangent_direction_basis(fb,fva,fold_model.semantic_group_vectors()); ft5=(fb@fb.index_select(0,ps.to(device)).T).topk(5,dim=1).values; ffeat=torch.stack(((fb*fva).sum(-1),(fva-fb).norm(dim=-1),text_resultant,ft5.mean(1)),dim=1); pu_device=pu.to(device); ff[pu_device]=tangent_transport(fb.index_select(0,pu_device),package["value"].to(device),gate(package["gate_features"].to(device))); generated=ccgr.generate_external(ff,fbas,ffeat); ff[pu_device]=generated.index_select(0,pu_device); ps_centroids=centroids.index_select(0,torch.searchsorted(seenclasses,ps)).to(device); fold_ridge=fit_ridge_attribute_map(ps_centroids,ps.to(device),attributes,float(config["ridge"])); fold_cra=AttributeResidualAlignment(fold_ridge,attributes,config["max_beta"]).to(device); fold_cra.raw_beta.data.copy_(main_cra.raw_beta.data); fold_cra.raw_beta.requires_grad_(False); fold_cra.eval()
            package["full_prototypes"]=ff; package["cra"]=fold_cra; package["pseudo_seen_mask"]=torch.isin(seenclasses,ps).to(device); packages.append(package); del fold_model
        calibrator=EpisodicBiasCalibration(config["max_gamma"]).to(device); optimizer=torch.optim.Adam(calibrator.parameters(),lr=float(config["lr"]),weight_decay=0.0); identity=SymmetricDiagonalMetric().to(device); identity.raw_log_weight.requires_grad_(False); mapping=torch.full((200,),-1,dtype=torch.long); mapping[seenclasses]=torch.arange(150); generators=[torch.Generator(device="cpu").manual_seed(seed*41000+i) for i in range(3)]; half=int(config["batch_half"]); history=[]; input_sha.update(h1.verify_inputs(base_config,paths,h1.OFFICIAL_KEYS)); tensors.update({name:torch.load(paths[name],map_location="cpu",weights_only=True) for name in h1.OFFICIAL_KEYS}); initial=evaluate(final_prototypes,parent.scale(),main_cra,calibrator,tensors,seenclasses,unseenclasses,device); best_H=initial["H"]; best_epoch=0; best_state=copy.deepcopy(calibrator.state_dict()); history.append({"epoch":0,"official_metrics_percent":initial,"gamma":0.0}); print(f"epoch=0 official_H={best_H:.6f} gamma=0")
        for epoch in range(1,int(config["epochs"])+1):
            loss_sum=0.0; count=0
            for fold_id,package in enumerate(packages):
                steps=min(package["seen_indices"].numel()//half,package["unseen_indices"].numel()//half)
                for _ in range(steps):
                    generator=generators[fold_id]; si=package["seen_indices"][torch.randperm(package["seen_indices"].numel(),generator=generator)[:half]]; ui=package["unseen_indices"][torch.randperm(package["unseen_indices"].numel(),generator=generator)[:half]]; indices=torch.cat((si,ui)); images=tensors["train_features"][indices].to(device).float(); targets=mapping[labels[indices]].to(device); logits=package["cra"].logits(images,package["full_prototypes"],package["scale"].to(device),identity,seenclasses); corrected=calibrator(logits,package["pseudo_seen_mask"]); loss=F.cross_entropy(corrected,targets); optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step(); loss_sum+=float(loss.detach()); count+=1
            official=evaluate(final_prototypes,parent.scale(),main_cra,calibrator,tensors,seenclasses,unseenclasses,device); row={"epoch":epoch,"loss":loss_sum/count,"official_metrics_percent":official,"gamma":float(calibrator.gamma().detach())}; history.append(row); print(f"epoch={epoch} loss={row['loss']:.6f} official_H={official['H']:.6f} gamma={row['gamma']:.6f}")
            if official["H"]>best_H: best_H=official["H"]; best_epoch=epoch; best_state=copy.deepcopy(calibrator.state_dict())
        calibrator.load_state_dict(best_state,strict=True); torch.save({"attempt_id":config["attempt_id"],"code_commit":code_commit,"config":config,"selected_epoch":best_epoch,"calibrator_state_dict":best_state,"history":history},output_dir/"ebc_model.pth"); parent_calibrator=EpisodicBiasCalibration(config["max_gamma"]).to(device); parent_metrics=evaluate(final_prototypes,parent.scale(),main_cra,parent_calibrator,tensors,seenclasses,unseenclasses,device); candidate_metrics=evaluate(final_prototypes,parent.scale(),main_cra,calibrator,tensors,seenclasses,unseenclasses,device); delta={key:candidate_metrics[key]-float(config["parent_metrics_percent"][key]) for key in ("U","S","H","ZS")}; gamma=float(calibrator.gamma().detach()); success=candidate_metrics["H"]>79.44820966336283 and delta["U"]>=-2 and delta["S"]>=-2 and abs(gamma)<0.98*float(config["max_gamma"]); atomic_write_json(output_dir/"data_fingerprints.json",{"files":input_sha}); metrics={"attempt_id":config["attempt_id"],"idea_id":config["idea_id"],"framework_id":config["framework_id"],"code_commit":code_commit,"config_sha256":config_sha,"base_config_sha256":base_config_sha,"evaluation_protocol":h1.EVALUATION_PROTOCOL,"test_used_for_selection":True,"unseen_images_used_for_gradient":False,"selected_epoch":best_epoch,"learned_gamma":gamma,"recomputed_parent_metrics_percent":parent_metrics,"parent_metrics_percent":config["parent_metrics_percent"],"candidate_metrics_percent":candidate_metrics,"delta_vs_parent_percent_points":delta,"success":success,"ebc_model_sha256":sha256_file(output_dir/"ebc_model.pth")}; atomic_write_json(output_dir/"metrics.json",metrics); print(metrics); return metrics
    finally:
        sys.stdout.flush(); sys.stdout=original_stdout; log_handle.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--config",type=Path,required=True); parser.add_argument("--output-dir",type=Path,required=True); parser.add_argument("--expected-commit",required=True); args=parser.parse_args(); run(args.config,args.output_dir,args.expected_commit)


if __name__=="__main__": main()
