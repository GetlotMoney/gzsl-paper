from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

from model.frameworks.v4.ccgr import ClassConditionedGeometricGenerator, tangent_direction_basis
from model.candidates.v2.modules.dpt import text_resultant_lengths
from model.candidates.v2.modules.edc import EpisodicDomainCompetition
from model.frameworks.v4.tg import VariableClassTGVPR, fixed_class_folds
from model.candidates.v2.trainers.train_elpt import _candidate_prototypes, _fold_package, _load_fold_checkpoint
from model.frameworks.v4.tst import TangentStepGate, tangent_transport
from model.frameworks.v2 import train as h1
from tools.reproducibility import configure_reproducibility
from tools.run_contract import atomic_write_json, current_code_commit, prepare_output_dir, require_clean_code_tree
from tools.runtime import sha256_file


CONFIG_KEYS={"schema_version","attempt_id","idea_id","framework_id","base_config","base_checkpoint","base_checkpoint_sha256","ntr_gate_model","ntr_gate_model_sha256","ccgr_model","ccgr_model_sha256","fold_checkpoint_dir","seed","epochs","batch_half","lr","weight_decay","max_correction","parent_metrics_percent"}


class TeeStream:
    def __init__(self,*streams): self.streams=streams
    def write(self,value):
        for stream in self.streams: stream.write(value)
        return len(value)
    def flush(self):
        for stream in self.streams: stream.flush()


def load_config(path:Path):
    path=path.resolve(); config=yaml.safe_load(path.read_text(encoding="utf-8")); actual=set(config) if isinstance(config,dict) else set()
    if not isinstance(config,dict) or actual!=CONFIG_KEYS: raise ValueError(f"EDC配置字段错误；缺少={sorted(CONFIG_KEYS-actual)}，多出={sorted(actual-CONFIG_KEYS)}。")
    if config["schema_version"] not in ("gzsl-paper.edc.v1","gzsl-paper.edc.v2") or config["attempt_id"] not in ("V2-TRY-070","V2-TRY-071") or config["idea_id"]!="IDEA-020": raise ValueError("EDC首次TRY身份错误。")
    expected_max=0.2 if config["attempt_id"]=="V2-TRY-070" else 0.05
    if int(config["epochs"])!=20 or int(config["batch_half"])!=32 or float(config["lr"])!=0.001 or float(config["weight_decay"])!=0.0001 or float(config["max_correction"])!=expected_max: raise ValueError("EDC训练参数错误。")
    if set(config["parent_metrics_percent"])!={"U","S","H","ZS"}: raise ValueError("EDC父指标不完整。")
    return config,sha256_file(path)


def _load_gate(config,device):
    path=Path(config["ntr_gate_model"])
    if sha256_file(path)!=config["ntr_gate_model_sha256"]: raise ValueError("EDC父NTR gate SHA不匹配。")
    payload=torch.load(path,map_location="cpu",weights_only=False); gate=TangentStepGate(input_dim=8,max_step=1.5); gate.load_state_dict(payload["gate_state_dict"],strict=True)
    for p in gate.parameters(): p.requires_grad_(False)
    return gate.to(device).eval()


@torch.no_grad()
def evaluate(prototypes,scale,edc,tensors,seenclasses,unseenclasses,device):
    allclasses=torch.arange(200); seen_mask=torch.isin(allclasses,seenclasses).to(device); unseen_mask=~seen_mask
    def pred(features,class_ids=None):
        p=prototypes if class_ids is None else prototypes.index_select(0,class_ids.to(device)); logits=F.normalize(features.to(device).float(),dim=-1)@p.T*scale
        if class_ids is None: logits=edc(logits,seen_mask,unseen_mask)
        result=logits.argmax(dim=1).cpu(); return class_ids[result] if class_ids is not None else result
    sp=pred(tensors["seen_features"]); up=pred(tensors["unseen_features"]); zp=pred(tensors["unseen_features"],unseenclasses); s=h1.per_class_accuracy(tensors["seen_labels"],sp,seenclasses); u=h1.per_class_accuracy(tensors["unseen_labels"],up,unseenclasses); z=h1.per_class_accuracy(tensors["unseen_labels"],zp,unseenclasses); H=2*s*u/(s+u) if s+u else 0.0; return {"U":u*100,"S":s*100,"H":H*100,"ZS":z*100}


def run(config_path:Path,output_dir:Path,expected_commit:str):
    require_clean_code_tree(); code_commit=current_code_commit()
    if code_commit!=expected_commit: raise ValueError("expected-commit与当前HEAD不一致。")
    config,config_sha=load_config(config_path); base_path=Path(config["base_config"])
    if not base_path.is_absolute(): base_path=Path.cwd()/base_path
    base_config,base_config_sha=h1.load_config(base_path); paths=h1.resolve_paths(base_config); input_sha=h1.verify_inputs(base_config,paths,h1.TRAINING_KEYS); checkpoint_path=Path(config["base_checkpoint"])
    if sha256_file(checkpoint_path)!=config["base_checkpoint_sha256"] or sha256_file(Path(config["ccgr_model"]))!=config["ccgr_model_sha256"]: raise ValueError("EDC父模型SHA不匹配。")
    device=torch.device(base_config["device"])
    if device.type!="cuda" or not torch.cuda.is_available(): raise RuntimeError("EDC要求CUDA。")
    output_dir=prepare_output_dir(output_dir)
    with (output_dir/"config.snapshot.yaml").open("x",encoding="utf-8") as handle: yaml.safe_dump(config,handle,allow_unicode=True,sort_keys=False)
    log_handle=(output_dir/"training.log").open("x",encoding="utf-8",buffering=1); original_stdout=sys.stdout; sys.stdout=TeeStream(sys.stdout,log_handle)
    try:
        seed=int(config["seed"]); configure_reproducibility(seed,strict_determinism=True,deterministic_warn_only=False); tensors={name:torch.load(paths[name],map_location="cpu",weights_only=True) for name in ("sentence_embeds","train_features","train_labels")}; labels=tensors["train_labels"].long(); seenclasses=torch.unique(labels,sorted=True); allclasses=torch.arange(200); unseenclasses=allclasses[~torch.isin(allclasses,seenclasses)]; checkpoint=torch.load(checkpoint_path,map_location="cpu",weights_only=False); centroids=h1.visual_centroids(tensors["train_features"],labels,seenclasses); parent=VariableClassTGVPR(tensors["sentence_embeds"],seenclasses,centroids,dropout=base_config["dropout"],inner_ratio=base_config["inner_ratio"],outer_ratio=base_config["outer_ratio"],temperature=base_config["temperature"]); parent.load_state_dict(checkpoint["model_state_dict"],strict=True); parent=parent.to(device).eval(); gate=_load_gate(config,device); folds=fixed_class_folds(seenclasses); ntr,_=_candidate_prototypes(parent,gate,seenclasses,unseenclasses,device,"top5_vector",folds,"tangent"); base=parent.base_prototypes(); value=parent.value_candidate(allclasses.to(device)); roles=parent.semantic_group_vectors(); basis=tangent_direction_basis(base,value,roles); sim=base@base.index_select(0,seenclasses.to(device)).T; top5=sim.topk(5,dim=1).values; text_resultant=text_resultant_lengths(tensors["sentence_embeds"]).to(device); features=torch.stack(((base*value).sum(-1),(value-base).norm(dim=-1),text_resultant,top5.mean(1)),dim=1); ccgr=ClassConditionedGeometricGenerator(ntr,basis,features,unseenclasses,parent.scale(),hidden_dim=32,max_magnitude=0.2,initial_magnitude=0.02).to(device); cp=torch.load(Path(config["ccgr_model"]),map_location="cpu",weights_only=False); ccgr.load_state_dict(cp["model_state_dict"],strict=True); ccgr.eval(); final_prototypes=ccgr.prototypes().detach(); packages=[]
        for fold_id,(ps,pu) in enumerate(folds):
            fold_model=_load_fold_checkpoint(fold_id,ps,tensors["sentence_embeds"],tensors["train_features"],labels,base_config,device,config["fold_checkpoint_dir"]); package=_fold_package(fold_model,ps,pu,tensors,seenclasses,device,"top5_vector")
            with torch.no_grad():
                fb=package["base_all"].to(device); ff=package["fold_full"].to(device).clone(); fva=fold_model.value_candidate(allclasses.to(device)); fr=fold_model.semantic_group_vectors(); fbas=tangent_direction_basis(fb,fva,fr); support=fb.index_select(0,ps.to(device)); ft5=(fb@support.T).topk(5,dim=1).values; ffeat=torch.stack(((fb*fva).sum(-1),(fva-fb).norm(dim=-1),text_resultant,ft5.mean(1)),dim=1); step=gate(package["gate_features"].to(device)); pu_d=pu.to(device); ff[pu_d]=tangent_transport(fb.index_select(0,pu_d),package["value"].to(device),step); generated=ccgr.generate_external(ff,fbas,ffeat); ff[pu_d]=generated.index_select(0,pu_d)
            package["competition"]=ff.index_select(0,seenclasses.to(device)); package["pseudo_seen_mask"]=torch.isin(seenclasses,ps).to(device); package["pseudo_unseen_mask"]=torch.isin(seenclasses,pu).to(device); packages.append(package); del fold_model
        edc=EpisodicDomainCompetition(config["max_correction"]).to(device); optimizer=torch.optim.Adam(edc.parameters(),lr=float(config["lr"]),weight_decay=float(config["weight_decay"])); mapping=torch.full((200,),-1,dtype=torch.long); mapping[seenclasses]=torch.arange(150); generators=[torch.Generator(device="cpu").manual_seed(seed*21000+i) for i in range(3)]; half=int(config["batch_half"]); history=[]
        for epoch in range(1,int(config["epochs"])+1):
            loss_sum=0.0; count=0; corrections=[]
            for fold_id,package in enumerate(packages):
                steps=min(package["seen_indices"].numel()//half,package["unseen_indices"].numel()//half)
                for _ in range(steps):
                    g=generators[fold_id]; si=package["seen_indices"][torch.randperm(package["seen_indices"].numel(),generator=g)[:half]]; ui=package["unseen_indices"][torch.randperm(package["unseen_indices"].numel(),generator=g)[:half]]; indices=torch.cat((si,ui)); images=tensors["train_features"][indices].to(device).float(); targets=mapping[labels[indices]].to(device); logits=F.normalize(images,dim=-1)@package["competition"].T*package["scale"].to(device); corrected=edc(logits,package["pseudo_seen_mask"],package["pseudo_unseen_mask"]); loss=F.cross_entropy(corrected,targets); optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step(); loss_sum+=float(loss.detach()); count+=1; corrections.append(edc.correction(logits.detach(),package["pseudo_seen_mask"],package["pseudo_unseen_mask"]).detach().cpu())
            c=torch.cat(corrections); row={"epoch":epoch,"loss":loss_sum/count,"correction_mean":float(c.mean()),"correction_std":float(c.std(unbiased=False)),"correction_min":float(c.min()),"correction_max":float(c.max())}; history.append(row); print(f"epoch={epoch} loss={row['loss']:.6f} correction={row['correction_mean']:.6f}+/-{row['correction_std']:.6f}")
        torch.save({"attempt_id":config["attempt_id"],"code_commit":code_commit,"config":config,"edc_state_dict":copy.deepcopy(edc.state_dict()),"history":history},output_dir/"edc_model.pth")
        # official test严格在EDC训练结束后加载。
        input_sha.update(h1.verify_inputs(base_config,paths,h1.OFFICIAL_KEYS)); tensors.update({name:torch.load(paths[name],map_location="cpu",weights_only=True) for name in h1.OFFICIAL_KEYS}); parent_edc=EpisodicDomainCompetition(config["max_correction"]).to(device); parent_metrics=evaluate(final_prototypes,parent.scale(),parent_edc,tensors,seenclasses,unseenclasses,device); candidate_metrics=evaluate(final_prototypes,parent.scale(),edc,tensors,seenclasses,unseenclasses,device); delta={key:candidate_metrics[key]-float(config["parent_metrics_percent"][key]) for key in ("U","S","H","ZS")}; all_logits=torch.cat((F.normalize(tensors["seen_features"].to(device).float(),dim=-1)@final_prototypes.T*parent.scale(),F.normalize(tensors["unseen_features"].to(device).float(),dim=-1)@final_prototypes.T*parent.scale())); sm=torch.isin(allclasses,seenclasses).to(device); um=~sm; corr=edc.correction(all_logits,sm,um).detach(); stats={"mean":float(corr.mean()),"std":float(corr.std(unbiased=False)),"min":float(corr.min()),"max":float(corr.max())}; success=delta["H"]>=0.20 and delta["U"]>=-2 and delta["S"]>=-2 and stats["std"]>0.01 and max(abs(stats["min"]),abs(stats["max"]))<0.196; atomic_write_json(output_dir/"data_fingerprints.json",{"files":input_sha}); metrics={"attempt_id":config["attempt_id"],"idea_id":config["idea_id"],"framework_id":config["framework_id"],"code_commit":code_commit,"config_sha256":config_sha,"base_config_sha256":base_config_sha,"evaluation_protocol":h1.EVALUATION_PROTOCOL,"test_used_for_selection":True,"unseen_images_used_for_gradient":False,"recomputed_parent_metrics_percent":parent_metrics,"parent_metrics_percent":config["parent_metrics_percent"],"candidate_metrics_percent":candidate_metrics,"delta_vs_parent_percent_points":delta,"correction_stats":stats,"success":success,"edc_model_sha256":sha256_file(output_dir/"edc_model.pth")}; atomic_write_json(output_dir/"metrics.json",metrics); print(metrics); return metrics
    finally:
        sys.stdout.flush(); sys.stdout=original_stdout; log_handle.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--config",type=Path,required=True); parser.add_argument("--output-dir",type=Path,required=True); parser.add_argument("--expected-commit",required=True); args=parser.parse_args(); run(args.config,args.output_dir,args.expected_commit)


if __name__=="__main__": main()
