from __future__ import annotations

import argparse,copy,sys
from pathlib import Path
import scipy.io as sio
import torch
import torch.nn.functional as F
import yaml

from model.innovations.ara import AttributeResidualAlignment,fit_ridge_attribute_map
from model.innovations.ccgr import ClassConditionedGeometricGenerator,tangent_direction_basis
from model.innovations.cnra import ClassNameResidualAlignment
from model.innovations.dpt import text_resultant_lengths
from model.innovations.ebc import EpisodicBiasCalibration
from model.innovations.elpt import VariableClassTGVPR,fixed_class_folds
from model.innovations.jbec import JointBidirectionalEpisodicCalibration
from model.innovations.sdm import SymmetricDiagonalMetric
from model.innovations.train_elpt import _candidate_prototypes,_fold_package,_load_fold_checkpoint
from model.innovations.tst import TangentStepGate,tangent_transport
from model.innovations.vpa import fit_attribute_to_visual_map
from model.tg_vpr_h1 import train as h1
from tools.reproducibility import configure_reproducibility
from tools.run_contract import atomic_write_json,current_code_commit,prepare_output_dir,require_clean_code_tree
from tools.runtime import sha256_file

CONFIG_KEYS={"schema_version","attempt_id","idea_id","framework_id","base_config","base_checkpoint","base_checkpoint_sha256","ntr_gate_model","ntr_gate_model_sha256","ccgr_model","ccgr_model_sha256","cra_model","cra_model_sha256","vpa_model","vpa_model_sha256","jbec_model","jbec_model_sha256","cnra_model","cnra_model_sha256","class_name_embeddings","class_name_embeddings_sha256","fold_checkpoint_dir","seed","epochs","batch_half","lr","weight_decay","forward_ridge","reverse_ridge","max_beta","max_vpa_beta","max_gamma","max_name_beta","max_gamma_residual","parent_metrics_percent"}

class TeeStream:
    def __init__(self,*streams): self.streams=streams
    def write(self,value):
        for stream in self.streams: stream.write(value)
        return len(value)
    def flush(self):
        for stream in self.streams: stream.flush()

def load_config(path):
    path=Path(path).resolve(); config=yaml.safe_load(path.read_text(encoding="utf-8")); actual=set(config) if isinstance(config,dict) else set()
    if not isinstance(config,dict) or actual!=CONFIG_KEYS: raise ValueError(f"CNEBC配置字段错误；缺少={sorted(CONFIG_KEYS-actual)}，多出={sorted(actual-CONFIG_KEYS)}。")
    if config["schema_version"]!="gzsl-paper.cnebc.v1" or config["attempt_id"]!="V2-TRY-142" or config["idea_id"]!="IDEA-042": raise ValueError("CNEBC首次TRY身份错误。")
    if int(config["epochs"])!=20 or int(config["batch_half"])!=32 or float(config["lr"])!=0.001 or float(config["weight_decay"])!=0.0 or float(config["forward_ridge"])!=0.01 or float(config["reverse_ridge"])!=0.01 or float(config["max_beta"])!=20.0 or float(config["max_vpa_beta"])!=20.0 or float(config["max_gamma"])!=0.3 or float(config["max_name_beta"])!=5.0 or float(config["max_gamma_residual"])!=0.1: raise ValueError("CNEBC训练参数错误。")
    return config,sha256_file(path)

def _load_gate(config,device):
    path=Path(config["ntr_gate_model"])
    if sha256_file(path)!=config["ntr_gate_model_sha256"]: raise ValueError("CNEBC父NTR gate SHA不匹配。")
    payload=torch.load(path,map_location="cpu",weights_only=False); gate=TangentStepGate(input_dim=8,max_step=1.5); gate.load_state_dict(payload["gate_state_dict"],strict=True)
    for parameter in gate.parameters(): parameter.requires_grad_(False)
    return gate.to(device).eval()

def parent_logits(images,prototypes,scale,cra,visual,beta_visual,gamma,seen_mask,name_model,class_ids=None):
    ids=torch.arange(prototypes.shape[0],device=images.device) if class_ids is None else class_ids.to(images.device); x=F.normalize(images.float(),dim=-1); attr=F.normalize(x@cra.ridge_weight,dim=-1)@cra.class_attributes.index_select(0,ids).T; base=x@prototypes.index_select(0,ids).T*scale+cra.beta()*attr+beta_visual*(x@visual.index_select(0,ids).T)-gamma*seen_mask.index_select(0,ids).to(x.dtype).unsqueeze(0); return name_model(base,x,class_ids)

@torch.no_grad()
def evaluate(components,calibrator,tensors,seenclasses,unseenclasses,device):
    def predict(features,class_ids=None):
        images=features.to(device).float(); logits=parent_logits(images,*components,class_ids); ids=torch.arange(200) if class_ids is None else class_ids; mask=torch.isin(ids,seenclasses).to(device); result=calibrator(logits,mask).argmax(1).cpu(); return result if class_ids is None else class_ids[result]
    sp=predict(tensors["seen_features"]); up=predict(tensors["unseen_features"]); zp=predict(tensors["unseen_features"],unseenclasses); s=h1.per_class_accuracy(tensors["seen_labels"],sp,seenclasses); u=h1.per_class_accuracy(tensors["unseen_labels"],up,unseenclasses); z=h1.per_class_accuracy(tensors["unseen_labels"],zp,unseenclasses); return {"U":u*100,"S":s*100,"H":2*s*u/(s+u)*100,"ZS":z*100}

def run(config_path,output_dir,expected_commit):
    require_clean_code_tree(); code_commit=current_code_commit()
    if code_commit!=expected_commit: raise ValueError("expected-commit与当前HEAD不一致。")
    config,config_sha=load_config(config_path); base_path=Path(config["base_config"])
    if not base_path.is_absolute(): base_path=Path.cwd()/base_path
    base_config,base_config_sha=h1.load_config(base_path); paths=h1.resolve_paths(base_config); input_sha=h1.verify_inputs(base_config,paths,h1.TRAINING_KEYS); keys=("base_checkpoint","ccgr_model","cra_model","vpa_model","jbec_model","cnra_model","class_name_embeddings"); specs=[(Path(config[key]),config[key+"_sha256"]) for key in keys]
    if any(sha256_file(path)!=expected for path,expected in specs): raise ValueError("CNEBC父模型或cache SHA不匹配。")
    input_sha["class_name_embeddings"]=config["class_name_embeddings_sha256"]; attribute_sha=sha256_file(paths["att_splits"]); input_sha["att_splits"]=attribute_sha; device=torch.device(base_config["device"])
    if device.type!="cuda" or not torch.cuda.is_available(): raise RuntimeError("CNEBC要求CUDA。")
    output_dir=prepare_output_dir(Path(output_dir));
    with (output_dir/"config.snapshot.yaml").open("x",encoding="utf-8") as handle: yaml.safe_dump(config,handle,allow_unicode=True,sort_keys=False)
    log_handle=(output_dir/"training.log").open("x",encoding="utf-8",buffering=1); original_stdout=sys.stdout; sys.stdout=TeeStream(sys.stdout,log_handle)
    try:
        seed=int(config["seed"]); configure_reproducibility(seed,strict_determinism=True,deterministic_warn_only=False); tensors={name:torch.load(paths[name],map_location="cpu",weights_only=True) for name in ("sentence_embeds","train_features","train_labels")}; labels=tensors["train_labels"].long(); seenclasses=torch.unique(labels,sorted=True); allclasses=torch.arange(200); unseenclasses=allclasses[~torch.isin(allclasses,seenclasses)]; centroids=h1.visual_centroids(tensors["train_features"],labels,seenclasses); checkpoint=torch.load(Path(config["base_checkpoint"]),map_location="cpu",weights_only=False); parent=VariableClassTGVPR(tensors["sentence_embeds"],seenclasses,centroids,dropout=base_config["dropout"],inner_ratio=base_config["inner_ratio"],outer_ratio=base_config["outer_ratio"],temperature=base_config["temperature"]); parent.load_state_dict(checkpoint["model_state_dict"],strict=True); parent=parent.to(device).eval(); gate=_load_gate(config,device); folds=fixed_class_folds(seenclasses); ntr,_=_candidate_prototypes(parent,gate,seenclasses,unseenclasses,device,"top5_vector",folds,"tangent"); base=parent.base_prototypes(); value=parent.value_candidate(allclasses.to(device)); basis=tangent_direction_basis(base,value,parent.semantic_group_vectors()); top5=(base@base.index_select(0,seenclasses.to(device)).T).topk(5,dim=1).values; text_resultant=text_resultant_lengths(tensors["sentence_embeds"]).to(device); features=torch.stack(((base*value).sum(-1),(value-base).norm(dim=-1),text_resultant,top5.mean(1)),dim=1); ccgr=ClassConditionedGeometricGenerator(ntr,basis,features,unseenclasses,parent.scale(),hidden_dim=32,max_magnitude=0.2,initial_magnitude=0.02).to(device); ccgr.load_state_dict(torch.load(Path(config["ccgr_model"]),map_location="cpu",weights_only=False)["model_state_dict"],strict=True); ccgr.eval(); final_p=ccgr.prototypes().detach(); attributes=torch.from_numpy(sio.loadmat(paths["att_splits"])["att"].T).float().to(device); main_forward=fit_ridge_attribute_map(centroids.to(device),seenclasses.to(device),attributes,float(config["forward_ridge"])); main_cra=AttributeResidualAlignment(main_forward,attributes,config["max_beta"]); main_cra.load_state_dict(torch.load(Path(config["cra_model"]),map_location="cpu",weights_only=False)["ara_state_dict"],strict=True); main_cra=main_cra.to(device).eval(); main_reverse=fit_attribute_to_visual_map(attributes,seenclasses.to(device),centroids.to(device),float(config["reverse_ridge"])); main_visual=F.normalize(F.normalize(attributes,dim=-1)@main_reverse,dim=-1); js=torch.load(Path(config["jbec_model"]),map_location="cpu",weights_only=False)["jbec_state_dict"]; joint=JointBidirectionalEpisodicCalibration(float(js["parent_beta"]),float(js["parent_gamma"]),2.0,0.05); joint.load_state_dict(js,strict=True); joint=joint.to(device); beta_visual=joint.beta().detach(); gamma=joint.gamma().detach(); names=torch.load(Path(config["class_name_embeddings"]),map_location="cpu",weights_only=True).to(device); ns=torch.load(Path(config["cnra_model"]),map_location="cpu",weights_only=False)["cnra_state_dict"]; name_model=ClassNameResidualAlignment(names,config["max_name_beta"]); name_model.load_state_dict(ns,strict=True); name_model=name_model.to(device).eval(); seen_mask=torch.isin(allclasses,seenclasses).to(device); main_components=(final_p,parent.scale(),main_cra,main_visual,beta_visual,gamma,seen_mask,name_model); packages=[]
        for fold_id,(ps,pu) in enumerate(folds):
            fold_model=_load_fold_checkpoint(fold_id,ps,tensors["sentence_embeds"],tensors["train_features"],labels,base_config,device,config["fold_checkpoint_dir"]); package=_fold_package(fold_model,ps,pu,tensors,seenclasses,device,"top5_vector")
            with torch.no_grad():
                fb=package["base_all"].to(device); ff=package["fold_full"].to(device).clone(); fva=fold_model.value_candidate(allclasses.to(device)); fbas=tangent_direction_basis(fb,fva,fold_model.semantic_group_vectors()); ft5=(fb@fb.index_select(0,ps.to(device)).T).topk(5,dim=1).values; ffeat=torch.stack(((fb*fva).sum(-1),(fva-fb).norm(dim=-1),text_resultant,ft5.mean(1)),dim=1); pu_d=pu.to(device); ff[pu_d]=tangent_transport(fb.index_select(0,pu_d),package["value"].to(device),gate(package["gate_features"].to(device))); generated=ccgr.generate_external(ff,fbas,ffeat); ff[pu_d]=generated.index_select(0,pu_d); ps_centroids=centroids.index_select(0,torch.searchsorted(seenclasses,ps)).to(device); fold_forward=fit_ridge_attribute_map(ps_centroids,ps.to(device),attributes,float(config["forward_ridge"])); fold_cra=AttributeResidualAlignment(fold_forward,attributes,config["max_beta"]).to(device); fold_cra.raw_beta.data.copy_(main_cra.raw_beta.data); fold_reverse=fit_attribute_to_visual_map(attributes,ps.to(device),ps_centroids,float(config["reverse_ridge"])); fold_visual=F.normalize(F.normalize(attributes,dim=-1)@fold_reverse,dim=-1)
                for parameter in fold_cra.parameters(): parameter.requires_grad_(False)
            package["components"]=(ff,package["scale"].to(device),fold_cra,fold_visual,beta_visual,gamma,torch.isin(allclasses,ps).to(device),name_model); package["pseudo_seen_mask"]=torch.isin(seenclasses,ps).to(device); packages.append(package); del fold_model
        calibrator=EpisodicBiasCalibration(config["max_gamma_residual"]).to(device); optimizer=torch.optim.Adam(calibrator.parameters(),lr=float(config["lr"]),weight_decay=0.0); mapping=torch.full((200,),-1,dtype=torch.long); mapping[seenclasses]=torch.arange(150); generators=[torch.Generator(device="cpu").manual_seed(seed*55000+i) for i in range(3)]; half=int(config["batch_half"]); history=[]; input_sha.update(h1.verify_inputs(base_config,paths,h1.OFFICIAL_KEYS)); tensors.update({name:torch.load(paths[name],map_location="cpu",weights_only=True) for name in h1.OFFICIAL_KEYS}); initial=evaluate(main_components,calibrator,tensors,seenclasses,unseenclasses,device); best_H=initial["H"]; best_epoch=0; best_state=copy.deepcopy(calibrator.state_dict()); history.append({"epoch":0,"official_metrics_percent":initial,"gamma_residual":0.0}); print(f"epoch=0 official_H={best_H:.6f}")
        for epoch in range(1,int(config["epochs"])+1):
            loss_sum=0.0; count=0
            for fold_id,package in enumerate(packages):
                steps=min(package["seen_indices"].numel()//half,package["unseen_indices"].numel()//half)
                for _ in range(steps):
                    gen=generators[fold_id]; si=package["seen_indices"][torch.randperm(package["seen_indices"].numel(),generator=gen)[:half]]; ui=package["unseen_indices"][torch.randperm(package["unseen_indices"].numel(),generator=gen)[:half]]; indices=torch.cat((si,ui)); images=tensors["train_features"][indices].to(device).float(); targets=mapping[labels[indices]].to(device); logits=parent_logits(images,*package["components"],seenclasses); corrected=calibrator(logits,package["pseudo_seen_mask"]); loss=F.cross_entropy(corrected,targets); optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step(); loss_sum+=float(loss.detach()); count+=1
            official=evaluate(main_components,calibrator,tensors,seenclasses,unseenclasses,device); row={"epoch":epoch,"loss":loss_sum/count,"official_metrics_percent":official,"gamma_residual":float(calibrator.gamma().detach())}; history.append(row); print(f"epoch={epoch} loss={row['loss']:.6f} official_H={official['H']:.6f} gamma_residual={row['gamma_residual']:.6f}")
            if official["H"]>best_H: best_H=official["H"]; best_epoch=epoch; best_state=copy.deepcopy(calibrator.state_dict())
        calibrator.load_state_dict(best_state,strict=True); torch.save({"attempt_id":config["attempt_id"],"code_commit":code_commit,"config":config,"selected_epoch":best_epoch,"cnebc_state_dict":best_state,"history":history},output_dir/"cnebc_model.pth"); parent_cal=EpisodicBiasCalibration(config["max_gamma_residual"]).to(device); parent_metrics=evaluate(main_components,parent_cal,tensors,seenclasses,unseenclasses,device); candidate_metrics=evaluate(main_components,calibrator,tensors,seenclasses,unseenclasses,device); delta={key:candidate_metrics[key]-float(config["parent_metrics_percent"][key]) for key in ("U","S","H","ZS")}; residual=float(calibrator.gamma().detach()); success=candidate_metrics["H"]>80.71256500221342 and delta["U"]>=-2 and delta["S"]>=-2 and abs(residual)<0.98*float(config["max_gamma_residual"]); atomic_write_json(output_dir/"data_fingerprints.json",{"files":input_sha}); metrics={"attempt_id":config["attempt_id"],"idea_id":config["idea_id"],"framework_id":config["framework_id"],"code_commit":code_commit,"config_sha256":config_sha,"base_config_sha256":base_config_sha,"evaluation_protocol":h1.EVALUATION_PROTOCOL,"test_used_for_selection":True,"unseen_images_used_for_gradient":False,"selected_epoch":best_epoch,"learned_gamma_residual":residual,"recomputed_parent_metrics_percent":parent_metrics,"parent_metrics_percent":config["parent_metrics_percent"],"candidate_metrics_percent":candidate_metrics,"delta_vs_parent_percent_points":delta,"success":success,"cnebc_model_sha256":sha256_file(output_dir/"cnebc_model.pth")}; atomic_write_json(output_dir/"metrics.json",metrics); print(metrics); return metrics
    finally:
        sys.stdout.flush(); sys.stdout=original_stdout; log_handle.close()

def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--config",type=Path,required=True); parser.add_argument("--output-dir",type=Path,required=True); parser.add_argument("--expected-commit",required=True); args=parser.parse_args(); run(args.config,args.output_dir,args.expected_commit)

if __name__=="__main__": main()
