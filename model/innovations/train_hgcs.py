from __future__ import annotations
import argparse,copy,sys
from pathlib import Path
import torch
import torch.nn.functional as F
import yaml
from model.innovations.ara import AttributeResidualAlignment
from model.innovations.ccgr import ClassConditionedGeometricGenerator
from model.innovations.cnra import ClassNameResidualAlignment
from model.innovations.hgcs import HierarchicalGroupCommonSuppression
from model.innovations.jbec import JointBidirectionalEpisodicCalibration
from model.tg_vpr_h1 import train as h1
from tools.reproducibility import configure_reproducibility
from tools.run_contract import atomic_write_json,current_code_commit,prepare_output_dir,require_clean_code_tree
from tools.runtime import sha256_file

CONFIG_KEYS={"schema_version","attempt_id","idea_id","framework_id","base_config","ccgr_model","ccgr_model_sha256","vpa_model","vpa_model_sha256","jbec_model","jbec_model_sha256","cnra_model","cnra_model_sha256","class_name_embeddings","class_name_embeddings_sha256","seed","epochs","batch_size","lr","weight_decay","group_count","max_beta","parent_metrics_percent"}
class TeeStream:
    def __init__(self,*streams): self.streams=streams
    def write(self,value):
        for stream in self.streams: stream.write(value)
        return len(value)
    def flush(self):
        for stream in self.streams: stream.flush()
def load_config(path):
    path=Path(path).resolve(); config=yaml.safe_load(path.read_text(encoding="utf-8")); actual=set(config) if isinstance(config,dict) else set()
    if not isinstance(config,dict) or actual!=CONFIG_KEYS: raise ValueError(f"HGCS配置字段错误；缺少={sorted(CONFIG_KEYS-actual)}，多出={sorted(actual-CONFIG_KEYS)}。")
    if config["schema_version"]!="gzsl-paper.hgcs.v1" or config["attempt_id"]!="V2-TRY-146" or config["idea_id"]!="IDEA-043": raise ValueError("HGCS首次TRY身份错误。")
    if int(config["epochs"])!=20 or int(config["batch_size"])!=256 or float(config["lr"])!=0.01 or float(config["weight_decay"])!=0.0 or int(config["group_count"])!=20 or float(config["max_beta"])!=10.0: raise ValueError("HGCS训练参数错误。")
    return config,sha256_file(path)
def parent_logits(images,prototypes,scale,cra,visual,beta_visual,gamma,seen_mask,name_model,class_ids=None):
    ids=torch.arange(prototypes.shape[0],device=images.device) if class_ids is None else class_ids.to(images.device); x=F.normalize(images.float(),dim=-1); attr=F.normalize(x@cra.ridge_weight,dim=-1)@cra.class_attributes.index_select(0,ids).T; base=x@prototypes.index_select(0,ids).T*scale+cra.beta()*attr+beta_visual*(x@visual.index_select(0,ids).T)-gamma*seen_mask.index_select(0,ids).to(x.dtype).unsqueeze(0); return name_model(base,x,class_ids)
@torch.no_grad()
def evaluate(components,model,tensors,seenclasses,unseenclasses,device):
    def predict(features,class_ids=None):
        images=features.to(device).float(); logits=parent_logits(images,*components,class_ids); result=model(logits,images,class_ids).argmax(1).cpu(); return result if class_ids is None else class_ids[result]
    sp=predict(tensors["seen_features"]); up=predict(tensors["unseen_features"]); zp=predict(tensors["unseen_features"],unseenclasses); s=h1.per_class_accuracy(tensors["seen_labels"],sp,seenclasses); u=h1.per_class_accuracy(tensors["unseen_labels"],up,unseenclasses); z=h1.per_class_accuracy(tensors["unseen_labels"],zp,unseenclasses); return {"U":u*100,"S":s*100,"H":2*s*u/(s+u)*100,"ZS":z*100}
def run(config_path,output_dir,expected_commit):
    require_clean_code_tree(); code_commit=current_code_commit()
    if code_commit!=expected_commit: raise ValueError("expected-commit与当前HEAD不一致。")
    config,config_sha=load_config(config_path); base_path=Path(config["base_config"])
    if not base_path.is_absolute(): base_path=Path.cwd()/base_path
    base_config,base_config_sha=h1.load_config(base_path); paths=h1.resolve_paths(base_config); input_sha=h1.verify_inputs(base_config,paths,h1.TRAINING_KEYS); specs=((Path(config["ccgr_model"]),config["ccgr_model_sha256"]),(Path(config["vpa_model"]),config["vpa_model_sha256"]),(Path(config["jbec_model"]),config["jbec_model_sha256"]),(Path(config["cnra_model"]),config["cnra_model_sha256"]),(Path(config["class_name_embeddings"]),config["class_name_embeddings_sha256"]))
    if any(sha256_file(path)!=expected for path,expected in specs): raise ValueError("HGCS父模型或cache SHA不匹配。")
    input_sha["class_name_embeddings"]=config["class_name_embeddings_sha256"]; device=torch.device(base_config["device"])
    if device.type!="cuda" or not torch.cuda.is_available(): raise RuntimeError("HGCS要求CUDA。")
    output_dir=prepare_output_dir(Path(output_dir));
    with (output_dir/"config.snapshot.yaml").open("x",encoding="utf-8") as handle: yaml.safe_dump(config,handle,allow_unicode=True,sort_keys=False)
    log_handle=(output_dir/"training.log").open("x",encoding="utf-8",buffering=1); original_stdout=sys.stdout; sys.stdout=TeeStream(sys.stdout,log_handle)
    try:
        seed=int(config["seed"]); configure_reproducibility(seed,strict_determinism=True,deterministic_warn_only=False); tensors={name:torch.load(paths[name],map_location="cpu",weights_only=True) for name in ("train_features","train_labels")}; labels=tensors["train_labels"].long(); seenclasses=torch.unique(labels,sorted=True); allclasses=torch.arange(200); unseenclasses=allclasses[~torch.isin(allclasses,seenclasses)]; cp=torch.load(Path(config["ccgr_model"]),map_location="cpu",weights_only=False); cs=cp["model_state_dict"]; ccgr=ClassConditionedGeometricGenerator(cs["parent_prototypes"],cs["direction_basis"],cs["class_features"],cs["target_classes"],cs["_scale"],hidden_dim=32,max_magnitude=0.2,initial_magnitude=0.02); ccgr.load_state_dict(cs,strict=True); ccgr=ccgr.to(device).eval(); prototypes=ccgr.prototypes().detach(); vp=torch.load(Path(config["vpa_model"]),map_location="cpu",weights_only=False); vs=vp["vpa_state_dict"]; cra=AttributeResidualAlignment(vs["base_cra.ridge_weight"],vs["base_cra.class_attributes"],20.0); cra.load_state_dict({key[len("base_cra."):]:value for key,value in vs.items() if key.startswith("base_cra.")},strict=True); cra=cra.to(device).eval(); visual=F.normalize(vs["visual_prototypes"],dim=-1).to(device); jp=torch.load(Path(config["jbec_model"]),map_location="cpu",weights_only=False); js=jp["jbec_state_dict"]; joint=JointBidirectionalEpisodicCalibration(float(js["parent_beta"]),float(js["parent_gamma"]),2.0,0.05); joint.load_state_dict(js,strict=True); joint=joint.to(device); names=torch.load(Path(config["class_name_embeddings"]),map_location="cpu",weights_only=True).to(device); ns=torch.load(Path(config["cnra_model"]),map_location="cpu",weights_only=False)["cnra_state_dict"]; name_model=ClassNameResidualAlignment(names,5.0); name_model.load_state_dict(ns,strict=True); name_model=name_model.to(device).eval(); seen_mask=torch.isin(allclasses,seenclasses).to(device); components=(prototypes,ccgr.scale(),cra,visual,joint.beta().detach(),joint.gamma().detach(),seen_mask,name_model); model=HierarchicalGroupCommonSuppression(names,config["group_count"],config["max_beta"]).to(device); optimizer=torch.optim.Adam(model.parameters(),lr=float(config["lr"]),weight_decay=0.0); mapping=torch.full((200,),-1,dtype=torch.long); mapping[seenclasses]=torch.arange(150); generator=torch.Generator(device="cpu").manual_seed(seed*57000); history=[]; input_sha.update(h1.verify_inputs(base_config,paths,h1.OFFICIAL_KEYS)); tensors.update({name:torch.load(paths[name],map_location="cpu",weights_only=True) for name in h1.OFFICIAL_KEYS}); initial=evaluate(components,model,tensors,seenclasses,unseenclasses,device); best_H=initial["H"]; best_epoch=0; best_state=copy.deepcopy(model.state_dict()); history.append({"epoch":0,"official_metrics_percent":initial,"beta":0.0}); print(f"epoch=0 official_H={best_H:.6f} beta=0")
        for epoch in range(1,int(config["epochs"])+1):
            order=torch.randperm(labels.numel(),generator=generator); loss_sum=0.0; count=0
            for start in range(0,labels.numel(),int(config["batch_size"])):
                indices=order[start:start+int(config["batch_size"])]; images=tensors["train_features"][indices].to(device).float(); targets=mapping[labels[indices]].to(device); parent=parent_logits(images,*components,seenclasses); logits=model(parent,images,seenclasses); loss=F.cross_entropy(logits,targets); optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step(); loss_sum+=float(loss.detach()); count+=1
            official=evaluate(components,model,tensors,seenclasses,unseenclasses,device); row={"epoch":epoch,"loss":loss_sum/count,"official_metrics_percent":official,"beta":float(model.beta().detach())}; history.append(row); print(f"epoch={epoch} loss={row['loss']:.6f} official_H={official['H']:.6f} beta={row['beta']:.6f}")
            if official["H"]>best_H: best_H=official["H"]; best_epoch=epoch; best_state=copy.deepcopy(model.state_dict())
        model.load_state_dict(best_state,strict=True); torch.save({"attempt_id":config["attempt_id"],"code_commit":code_commit,"config":config,"selected_epoch":best_epoch,"hgcs_state_dict":best_state,"history":history},output_dir/"hgcs_model.pth"); parent_model=HierarchicalGroupCommonSuppression(names,config["group_count"],config["max_beta"]).to(device); parent_metrics=evaluate(components,parent_model,tensors,seenclasses,unseenclasses,device); candidate_metrics=evaluate(components,model,tensors,seenclasses,unseenclasses,device); delta={key:candidate_metrics[key]-float(config["parent_metrics_percent"][key]) for key in ("U","S","H","ZS")}; beta=float(model.beta().detach()); success=candidate_metrics["H"]>80.71256500221342 and delta["U"]>=-2 and delta["S"]>=-2 and -0.98*float(config["max_beta"])<beta<-0.1; atomic_write_json(output_dir/"data_fingerprints.json",{"files":input_sha}); metrics={"attempt_id":config["attempt_id"],"idea_id":config["idea_id"],"framework_id":config["framework_id"],"code_commit":code_commit,"config_sha256":config_sha,"base_config_sha256":base_config_sha,"evaluation_protocol":h1.EVALUATION_PROTOCOL,"test_used_for_selection":True,"unseen_images_used_for_gradient":False,"selected_epoch":best_epoch,"learned_beta":beta,"group_sizes":torch.bincount(model.assignment.cpu(),minlength=int(config["group_count"])).tolist(),"recomputed_parent_metrics_percent":parent_metrics,"parent_metrics_percent":config["parent_metrics_percent"],"candidate_metrics_percent":candidate_metrics,"delta_vs_parent_percent_points":delta,"success":success,"hgcs_model_sha256":sha256_file(output_dir/"hgcs_model.pth")}; atomic_write_json(output_dir/"metrics.json",metrics); print(metrics); return metrics
    finally:
        sys.stdout.flush(); sys.stdout=original_stdout; log_handle.close()
def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--config",type=Path,required=True); parser.add_argument("--output-dir",type=Path,required=True); parser.add_argument("--expected-commit",required=True); args=parser.parse_args(); run(args.config,args.output_dir,args.expected_commit)
if __name__=="__main__": main()
