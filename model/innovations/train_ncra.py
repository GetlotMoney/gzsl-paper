from __future__ import annotations

import argparse, copy, sys
from pathlib import Path
import torch
import torch.nn.functional as F
import yaml

from model.innovations.cnra import ClassNameResidualAlignment
from model.innovations.train_chen_style import OFFICIAL_KEYS, random_batch_indices, resolve_paths, verify_inputs
from model.innovations.unified_seen import UnifiedSeenPrototypeModel
from model.tg_vpr_h1 import train as h1
from tools.cub_data import load_cub_split
from tools.reproducibility import configure_reproducibility
from tools.run_contract import atomic_torch_save, atomic_write_json, current_code_commit, prepare_output_dir, require_clean_code_tree, require_finite_gradients
from tools.runtime import sha256_file

EVALUATION_PROTOCOL="chen_shiming_code_aligned_test_selected_gzsl"
CONFIG_KEYS={"schema_version","experiment_id","idea_id","framework_id","dataset","evaluation_protocol","test_used_for_selection","unseen_images_used_for_gradient","strict_blind_claim","parent_model","parent_model_sha256","parent_metrics_percent","class_name_embeddings","class_name_embeddings_sha256","device","random_seed","batch_size","epochs","niters","report_interval","optimizer","learning_rate","weight_decay","max_beta","inputs","expected_sha256","class_order_sha256"}

def load_config(path):
    path=h1.repo_path(path); c=yaml.safe_load(path.read_text(encoding="utf-8")); actual=set(c) if isinstance(c,dict) else set()
    if not isinstance(c,dict) or actual!=CONFIG_KEYS: raise ValueError(f"NCRA配置字段错误；缺少={sorted(CONFIG_KEYS-actual)}，多出={sorted(actual-CONFIG_KEYS)}。")
    beta_by_schema={"gzsl-paper.ncra.v1":5.0,"gzsl-paper.ncra.v2":10.0,"gzsl-paper.ncra.v3":20.0}
    if c["schema_version"] not in beta_by_schema or c["experiment_id"]!="V2-INNOVATION-011" or c["idea_id"]!="IDEA-045": raise ValueError("NCRA身份错误。")
    if c["evaluation_protocol"]!=EVALUATION_PROTOCOL or c["test_used_for_selection"] is not True or c["unseen_images_used_for_gradient"] is not False or c["strict_blind_claim"] is not False: raise ValueError("NCRA协议边界错误。")
    if int(c["batch_size"])!=50 or int(c["epochs"])!=200 or int(c["niters"])!=28228 or int(c["report_interval"])!=141: raise ValueError("NCRA Chen训练量错误。")
    expected_beta=beta_by_schema[c["schema_version"]]
    if c["optimizer"]!="Adam" or float(c["learning_rate"])!=0.01 or float(c["weight_decay"])!=0.0 or float(c["max_beta"])!=expected_beta: raise ValueError("NCRA优化参数错误。")
    return c,sha256_file(path)

def evaluate(model,parent,names,tensors,seenclasses,unseenclasses,device):
    parent.eval(); model.eval(); prototypes=parent.prototypes()
    def predict(features,class_ids=None):
        ids=torch.arange(200,device=device) if class_ids is None else class_ids.to(device); images=features.to(device).float(); logits=F.normalize(images,dim=-1)@prototypes.index_select(0,ids).T*parent.scale(); pred=model(logits,images,ids).argmax(1).cpu(); return pred if class_ids is None else class_ids[pred]
    with torch.no_grad(): sp=predict(tensors["seen_features"]); up=predict(tensors["unseen_features"]); zp=predict(tensors["unseen_features"],unseenclasses)
    s=h1.per_class_accuracy(tensors["seen_labels"],sp,seenclasses); u=h1.per_class_accuracy(tensors["unseen_labels"],up,unseenclasses); z=h1.per_class_accuracy(tensors["unseen_labels"],zp,unseenclasses)
    return {"U":u*100,"S":s*100,"H":2*s*u/(s+u)*100,"ZS":z*100}

def run(config_path,output_dir,expected_commit,run_id):
    require_clean_code_tree(); commit=current_code_commit()
    if commit!=expected_commit: raise ValueError("expected-commit不一致。")
    config,config_sha=load_config(config_path); paths=resolve_paths(config); input_sha=verify_inputs(config,paths); parent_path=Path(config["parent_model"]); names_path=Path(config["class_name_embeddings"])
    if sha256_file(parent_path)!=config["parent_model_sha256"] or sha256_file(names_path)!=config["class_name_embeddings_sha256"]: raise ValueError("NCRA父模型或类名cache SHA错误。")
    device=torch.device(config["device"]); output_dir=prepare_output_dir(output_dir)
    with (output_dir/"config.snapshot.yaml").open("x",encoding="utf-8") as f: yaml.safe_dump(config,f,allow_unicode=True,sort_keys=False)
    log=(output_dir/"training.log").open("x",encoding="utf-8",buffering=1); old=sys.stdout; sys.stdout=h1.TeeStream(sys.stdout,log)
    try:
        seed=int(config["random_seed"]); rep=configure_reproducibility(seed,strict_determinism=True,deterministic_warn_only=False); sentence=torch.load(paths["sentence_embeds"],map_location="cpu",weights_only=True); features=torch.load(paths["train_features"],map_location="cpu",weights_only=True); labels=torch.load(paths["train_labels"],map_location="cpu",weights_only=True).long(); official={n:torch.load(paths[n],map_location="cpu",weights_only=True) for n in OFFICIAL_KEYS}; seen=torch.unique(labels,sorted=True); allc=torch.arange(200); unseen=allc[~torch.isin(allc,seen)]; cs,cu=load_cub_split(paths["res101"],paths["att_splits"],labels,official["seen_labels"],official["unseen_labels"],"cpu"); assert torch.equal(cs,seen) and torch.equal(cu,unseen)
        cent=h1.visual_centroids(features,labels,seen); payload=torch.load(parent_path,map_location="cpu",weights_only=False); pc=payload["config"]; parent=UnifiedSeenPrototypeModel(sentence,seen,cent,active_classes=allc,dropout=float(pc["dropout"]),inner_ratio=float(pc["inner_ratio"]),outer_ratio=float(pc["outer_ratio"]),temperature=float(pc["temperature"]),transport_hidden_dim=int(pc["transport_hidden_dim"]),generator_hidden_dim=int(pc["generator_hidden_dim"]),max_transport_step=float(pc["max_transport_step"]),max_generator_magnitude=float(pc["max_generator_magnitude"])).to(device); parent.load_state_dict(payload["model_state_dict"],strict=True); parent.eval(); [p.requires_grad_(False) for p in parent.parameters()]; prototypes=parent.prototypes().detach(); scale=parent.scale().detach(); names=torch.load(names_path,map_location="cpu",weights_only=True).to(device); model=ClassNameResidualAlignment(names,float(config["max_beta"])).to(device); optimizer=torch.optim.Adam(model.parameters(),lr=float(config["learning_rate"])); mapping=torch.full((200,),-1,dtype=torch.long); mapping[seen]=torch.arange(150); gen=torch.Generator().manual_seed(seed); history=[]; best_metrics=evaluate(model,parent,names,official,seen,unseen,device); best_h=best_metrics["H"]; best_state=copy.deepcopy(model.state_dict()); best_iter=-1; atomic_torch_save(output_dir/"model_best.pth",{"ncra_state_dict":best_state,"best_metrics_percent":best_metrics,"selected_iteration":best_iter,"config":config,"code_commit":commit,"reproducibility":rep})
        for i in range(int(config["niters"])):
            idx=random_batch_indices(labels.numel(),int(config["batch_size"]),gen); images=features[idx].to(device).float(); targets=mapping[labels[idx]].to(device); base=F.normalize(images,dim=-1)@prototypes.index_select(0,seen.to(device)).T*scale; logits=model(base,images,seen); loss=F.cross_entropy(logits,targets); optimizer.zero_grad(set_to_none=True); loss.backward(); require_finite_gradients(model); optimizer.step()
            if i%int(config["report_interval"])==0:
                metrics=evaluate(model,parent,names,official,seen,unseen,device); beta=float(model.beta().detach()); history.append({"iteration":i,"official_metrics_percent":metrics,"beta":beta,"loss":float(loss.detach())});
                if metrics["H"]>best_h: best_h=metrics["H"]; best_metrics=metrics; best_state=copy.deepcopy(model.state_dict()); best_iter=i; atomic_torch_save(output_dir/"model_best.pth",{"ncra_state_dict":best_state,"best_metrics_percent":best_metrics,"selected_iteration":best_iter,"config":config,"code_commit":commit,"reproducibility":rep})
                print(f"iter={i} H={metrics['H']:.6f} best_H={best_h:.6f} beta={beta:.6f}")
        atomic_torch_save(output_dir/"checkpoint_last.pth",{"ncra_state_dict":copy.deepcopy(model.state_dict()),"best_state_dict":best_state,"best_metrics_percent":best_metrics,"selected_iteration":best_iter,"history":history,"config":config,"code_commit":commit}); atomic_write_json(output_dir/"data_fingerprints.json",{"files":input_sha,"parent_model":config["parent_model_sha256"],"class_name_embeddings":config["class_name_embeddings_sha256"]}); beta=float(best_state["raw_beta"].tanh()*float(config["max_beta"])); metrics={"experiment_id":config["experiment_id"],"idea_id":config["idea_id"],"run_id":run_id,"code_commit":commit,"config_sha256":config_sha,"test_used_for_selection":True,"unseen_images_used_for_gradient":False,"parent_metrics_percent":config["parent_metrics_percent"],"best_metrics_percent":best_metrics,"delta_vs_parent_percent_points":{k:best_metrics[k]-float(config["parent_metrics_percent"][k]) for k in ("U","S","H","ZS")},"selected_iteration":best_iter,"learned_beta":beta,"official_test_evaluation_count":len(history)+1,"model_sha256":sha256_file(output_dir/"model_best.pth"),"checkpoint_last_sha256":sha256_file(output_dir/"checkpoint_last.pth")}; atomic_write_json(output_dir/"metrics.json",metrics); print(metrics); return metrics
    finally: sys.stdout.flush(); sys.stdout=old; log.close()

def main():
    p=argparse.ArgumentParser(); p.add_argument("--config",type=Path,required=True); p.add_argument("--output-dir",type=Path,required=True); p.add_argument("--expected-commit",required=True); p.add_argument("--run-id",required=True); a=p.parse_args(); run(a.config,a.output_dir,a.expected_commit,a.run_id)
if __name__=="__main__": main()
