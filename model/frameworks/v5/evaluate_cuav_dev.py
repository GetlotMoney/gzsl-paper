"""CUAV preliminary B=1 Gate with same-cost crop controls."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from model.frameworks.v5.cuav import CUAVModel, cuav_action_losses
from model.frameworks.v5.cuav_data import load_subset, validate_bundle
from tools.prepare_cuav_assets import raw_crop_with_box, WINDOWS
from tools.run_contract import atomic_write_json, current_code_commit, prepare_output_dir, require_clean_code_tree
from tools.runtime import sha256_file


SCHEMA = "gzsl-paper.v5-cuav-dev-eval.v1"


class SelectedCropDataset(Dataset):
    def __init__(self, paths, actions, preprocess):
        self.paths, self.actions, self.preprocess = paths, actions, preprocess

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as handle:
            crop, box = raw_crop_with_box(handle.convert("RGB"), WINDOWS[int(self.actions[index])])
            return self.preprocess(crop), torch.tensor(box), int(self.actions[index])


@torch.no_grad()
def encode_selected(model, preprocess, paths, actions, device, batch_size=32, workers=8):
    loader = DataLoader(
        SelectedCropDataset(paths, actions, preprocess), batch_size=batch_size,
        shuffle=False, num_workers=workers, pin_memory=True,
    )
    rows, boxes, recorded = [], [], []
    for images, current_boxes, current_actions in loader:
        rows.append(F.normalize(model.encode_image(images.to(device).float()).float(), dim=-1).cpu())
        boxes.append(current_boxes)
        recorded.append(current_actions)
    if not torch.equal(torch.cat(recorded).long(), actions.long()):
        raise RuntimeError("CUAV selected action执行顺序错误。")
    return torch.cat(rows), torch.cat(boxes)


@torch.no_grad()
def encode_lowres_selected(model, preprocessed, actions, device, batch_size=32):
    rows = []
    for start in range(0, len(actions), batch_size):
        end = min(start + batch_size, len(actions))
        source = preprocessed[start:end]
        images = (
            source.clone().float()
            if isinstance(source, torch.Tensor)
            else torch.from_numpy(np.asarray(source).copy()).float()
        )
        crops = []
        for local, action in enumerate(actions[start:end].tolist()):
            row, column = WINDOWS[int(action)]
            crop = images[local:local+1, :, row*14:(row+6)*14, column*14:(column+6)*14]
            crops.append(F.interpolate(crop, size=(336, 336), mode="bicubic", align_corners=False)[0])
        batch = torch.stack(crops).to(device)
        rows.append(F.normalize(model.encode_image(batch).float(), dim=-1).cpu())
    return torch.cat(rows)


def load_checkpoint(spec, expected_condition, expected_commit, bundle_id):
    path = Path(spec["path"])
    if not path.is_file() or sha256_file(path) != spec["sha256"]:
        raise ValueError("CUAV checkpoint SHA错误。")
    value = torch.load(path, map_location="cpu", weights_only=True)
    if (
        value.get("schema_version") != "gzsl-paper.v5-cuav-dev-train.v1"
        or value.get("condition_id") != expected_condition
        or value.get("semantic_off") != (expected_condition == "CUAV_IMAGE_ONLY")
        or value.get("code_commit") != expected_commit
        or spec["training_commit"] != expected_commit
        or value.get("bundle_id") != bundle_id
        or not all(value.get("gradient_receipt", {}).get(key) is True for key in (
            ("step1_Wa_nonzero", "step2_Wa_nonzero", "step2_Wz_nonzero", "step2_Wq_Ws_not_applicable")
            if expected_condition == "CUAV_IMAGE_ONLY"
            else ("step1_Wa_nonzero", "step2_Wa_nonzero", "step2_Wz_nonzero", "step2_Wq_nonzero", "step2_Ws_nonzero")
        ))
    ):
        raise ValueError("CUAV checkpoint身份错误。")
    return value


@torch.no_grad()
def policy_actions(checkpoint, values, device, *, semantic_off):
    model = CUAVModel(values["name_embeddings"], values["class_ids"]).to(device)
    model.visual_module.load_state_dict(checkpoint["policy_state_dict"], strict=True)
    model.eval(); model.reset_call_counts()
    actions, entropies = [], []
    for start in range(0, len(values["labels"]), 64):
        result = model.policy(values["full_cls"][start:start+64].to(device).float(), semantic_off=semantic_off)
        actions.append(result["action"].cpu())
        entropies.append((-(result["policy"] * torch.log(result["policy"].clamp_min(1e-6))).sum(1)).cpu())
    return torch.cat(actions), torch.cat(entropies), dict(model.call_counts)


@torch.no_grad()
def selected_logits(checkpoint, values, crop_features, device, *, semantic_off=False, interaction_off=False):
    model = CUAVModel(values["name_embeddings"], values["class_ids"]).to(device)
    model.visual_module.load_state_dict(checkpoint["policy_state_dict"], strict=True)
    model.eval(); model.reset_call_counts()
    outputs = []
    for start in range(0, len(values["labels"]), 64):
        result = model.selected_update(
            values["full_cls"][start:start+64].to(device).float(),
            crop_features[start:start+64].to(device).float(),
            semantic_off=semantic_off, interaction_off=interaction_off,
        )
        outputs.append(result["logits"].cpu())
    return torch.cat(outputs), dict(model.call_counts)


@torch.no_grad()
def static_best_action(train_values, train_crops, device):
    model = CUAVModel(train_values["name_embeddings"], train_values["class_ids"]).to(device).eval()
    class_map = torch.full((200,), -1, dtype=torch.long)
    class_map[train_values["class_ids"].long()] = torch.arange(train_values["class_ids"].numel())
    targets = class_map[train_values["labels"].long()]
    sums = torch.zeros(25, dtype=torch.double)
    for start in range(0, len(targets), 16):
        crops = torch.from_numpy(np.array(train_crops[start:start+16], copy=True)).to(device).float()
        output = model.training_forward(train_values["full_cls"][start:start+16].to(device).float(), crops)
        sums += cuav_action_losses(output, targets[start:start+16].to(device)).double().sum(0).cpu()
    means = sums / len(targets)
    return int(means.argmin()), [float(x) for x in means]


def metrics(logits, values):
    labels, class_ids = values["labels"].long(), values["class_ids"].long()
    prediction = class_ids[logits.argmax(1)]
    classes = torch.unique(labels, sorted=True)
    vector = torch.stack([prediction[labels.eq(c)].eq(c).double().mean() for c in classes])
    return {"macro_top1": 100*float(vector.mean()), "micro_top1": 100*float(prediction.eq(labels).double().mean()), "per_class": vector, "prediction": prediction}


def comparison(full, other, matrix):
    diff = 100*(full-other); samples = diff[matrix].mean(1)
    ci = torch.quantile(samples, torch.tensor([0.025, 0.975], dtype=torch.double))
    return {"observed_pp": float(diff.mean()), "ci95": [float(ci[0]), float(ci[1])]}


def run(config_path, output_path, expected_commit, expected_config_sha):
    require_clean_code_tree()
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")); config_sha = sha256_file(Path(config_path))
    required = {
        "schema_version", "experiment_id", "train_manifest", "train_manifest_sha256",
        "eval_manifest", "eval_manifest_sha256", "oracle_manifest", "oracle_manifest_sha256",
        "bundle_manifest", "bundle_manifest_sha256", "asset_generation_commit",
        "full_checkpoint", "image_only_checkpoint", "clip_checkpoint", "clip_checkpoint_sha256",
        "device", "bootstrap_seed", "bootstrap_samples", "unseen_images_used_for_gradient",
        "official_test_loaded", "pclr_online_inference",
    }
    if (
        not isinstance(config, dict) or set(config) != required
        or config["schema_version"] != SCHEMA or config_sha != expected_config_sha
        or current_code_commit() != expected_commit or int(config["bootstrap_seed"]) != 7
        or int(config["bootstrap_samples"]) != 10000
        or config["unseen_images_used_for_gradient"] is not False
        or config["official_test_loaded"] is not False or config["pclr_online_inference"] is not False
    ):
        raise ValueError("CUAV eval配置错误。")
    eval_values, _, preprocessed, paths, eval_meta = load_subset(
        config["eval_manifest"], config["eval_manifest_sha256"], subset="dev_eval",
        open_crops=False, open_paths=True, open_preprocessed=True,
    )
    train_values, train_crops, _, _, train_meta = load_subset(
        config["train_manifest"], config["train_manifest_sha256"], subset="dev_train",
        open_crops=True, open_paths=False,
    )
    bundle = validate_bundle(config["bundle_manifest"], config["bundle_manifest_sha256"], subset="dev_eval", subset_sha=config["eval_manifest_sha256"])
    validate_bundle(config["bundle_manifest"], config["bundle_manifest_sha256"], subset="dev_train", subset_sha=config["train_manifest_sha256"])
    validate_bundle(config["bundle_manifest"], config["bundle_manifest_sha256"], subset="dev_eval_oracle", subset_sha=config["oracle_manifest_sha256"])
    bundle_id = bundle["common_identity"]["bundle_id"]
    if (
        eval_meta["common_identity"]["code_commit"] != config["asset_generation_commit"]
        or train_meta["common_identity"]["bundle_id"] != bundle_id
        or eval_values["class_ids"].numel() != 150 or torch.unique(eval_values["labels"]).numel() != 50
        or not bool(torch.isin(eval_values["labels"].long(), eval_values["class_ids"].long()).all())
        or train_values["class_ids"].numel() != 100
        or len(train_values["labels"]) != 4702 or len(eval_values["labels"]) != 2355
    ):
        raise ValueError("CUAV 100/50资产边界错误。")
    full_cp = load_checkpoint(config["full_checkpoint"], "CUAV_FULL", expected_commit, bundle_id)
    image_cp = load_checkpoint(config["image_only_checkpoint"], "CUAV_IMAGE_ONLY", expected_commit, bundle_id)
    if not Path(config["clip_checkpoint"]).is_file() or sha256_file(Path(config["clip_checkpoint"])) != config["clip_checkpoint_sha256"]:
        raise ValueError("CUAV CLIP checkpoint错误。")
    import clip
    device = torch.device(config["device"])
    clip_model, preprocess = clip.load(config["clip_checkpoint"], device=device, jit=False); clip_model=clip_model.float().eval()
    full_actions, full_entropy, full_policy_calls = policy_actions(full_cp, eval_values, device, semantic_off=False)
    soff_actions, soff_entropy, soff_policy_calls = policy_actions(full_cp, eval_values, device, semantic_off=True)
    image_actions, image_entropy, image_policy_calls = policy_actions(image_cp, eval_values, device, semantic_off=True)
    static_action, static_losses = static_best_action(train_values, train_crops, device)
    center_actions = torch.full_like(full_actions, 12); static_actions = torch.full_like(full_actions, static_action)
    full_crop, full_boxes = encode_selected(clip_model, preprocess, paths, full_actions, device)
    soff_crop, _ = encode_selected(clip_model, preprocess, paths, soff_actions, device)
    image_crop, _ = encode_selected(clip_model, preprocess, paths, image_actions, device)
    center_crop, _ = encode_selected(clip_model, preprocess, paths, center_actions, device)
    static_crop, _ = encode_selected(clip_model, preprocess, paths, static_actions, device)
    lowres_selected = encode_lowres_selected(clip_model, preprocessed, full_actions, device)
    logits = {}
    logits["full"], full_update_calls = selected_logits(full_cp, eval_values, full_crop, device)
    logits["parent"] = CUAVModel(eval_values["name_embeddings"], eval_values["class_ids"]).semantic_module(eval_values["full_cls"])["parent_logits"]
    logits["s_off"], soff_update_calls = selected_logits(full_cp, eval_values, soff_crop, device, semantic_off=True)
    logits["v_off"], voff_calls = selected_logits(full_cp, eval_values, lowres_selected, device)
    logits["i_off"], ioff_calls = selected_logits(full_cp, eval_values, full_crop, device, interaction_off=True)
    logits["center"], _ = selected_logits(full_cp, eval_values, center_crop, device)
    logits["static_best"], _ = selected_logits(full_cp, eval_values, static_crop, device)
    logits["image_only"], image_update_calls = selected_logits(image_cp, eval_values, image_crop, device, semantic_off=True)
    expected_full_boxes = eval_values["crop_boxes"][torch.arange(len(full_actions)), full_actions]
    if not torch.equal(full_boxes.long(), expected_full_boxes.long()):
        raise ValueError("CUAV selected raw crop box与资产geometry不一致。")
    values = {name: metrics(value, eval_values) for name, value in logits.items()}
    generator = torch.Generator().manual_seed(7); matrix = torch.randint(0,50,(10000,50),generator=generator)
    comparisons = {name: comparison(values["full"]["per_class"], values[name]["per_class"], matrix) for name in ("parent","s_off","v_off","i_off","center","static_best","image_only")}
    labels=eval_values["labels"].long(); parent_pred=values["parent"]["prediction"]; full_pred=values["full"]["prediction"]
    corrected=full_pred.eq(labels)&parent_pred.ne(labels); damaged=full_pred.ne(labels)&parent_pred.eq(labels)
    histogram=torch.bincount(full_actions,minlength=25); highest=float(histogram.max())/len(full_actions)
    gates={
        name: comparisons[name]["observed_pp"]>=1 and comparisons[name]["ci95"][0]>0
        for name in comparisons
    }
    gates.update({"net_positive":int(corrected.sum()-damaged.sum())>0,"occupancy":highest<=0.70,"used_actions":int(histogram.gt(0).sum())>=10})
    preliminary=all(gates.values())
    # Oracle is opened only after all B=1 actions/logits are frozen.
    oracle_values, oracle_crops, _, _, oracle_meta = load_subset(
        config["oracle_manifest"], config["oracle_manifest_sha256"], subset="dev_eval_oracle",
        open_crops=True, open_paths=False,
    )
    if (
        oracle_meta["image_order_sha256"] != eval_meta["image_order_sha256"]
        or oracle_meta["common_identity"]["bundle_id"] != bundle_id
        or not torch.equal(oracle_values["class_ids"].long(), eval_values["class_ids"].long())
        or not torch.equal(oracle_values["labels"].long(), eval_values["labels"].long())
        or len(oracle_values["labels"]) != 2355
    ):
        raise ValueError("CUAV oracle rows与B1 eval不一致。")
    oracle_model=CUAVModel(oracle_values["name_embeddings"],oracle_values["class_ids"]).to(device).eval()
    oracle_predictions=[]
    class_map=torch.full((200,),-1,dtype=torch.long); class_map[oracle_values["class_ids"].long()]=torch.arange(150)
    target_pos=class_map[oracle_values["labels"].long()]
    for start in range(0,len(target_pos),16):
        crops=torch.from_numpy(np.array(oracle_crops[start:start+16],copy=True)).to(device).float()
        out=oracle_model.training_forward(oracle_values["full_cls"][start:start+16].to(device).float(),crops)
        targets=target_pos[start:start+16].to(device)
        action_loss=cuav_action_losses(out,targets); best=action_loss.argmin(1)
        selected=out["action_final_logits"][torch.arange(best.numel(),device=device),best]
        oracle_predictions.append(oracle_values["class_ids"][selected.argmax(1).cpu()])
    oracle_pred=torch.cat(oracle_predictions); oracle_classes=torch.unique(oracle_values["labels"],sorted=True)
    oracle_vector=torch.stack([oracle_pred[oracle_values["labels"].eq(c)].eq(c).double().mean() for c in oracle_classes])
    result={
        "schema_version":SCHEMA,"code_commit":expected_commit,"config_sha256":config_sha,"bundle_id":bundle_id,
        "metrics":{name:{"macro_top1":v["macro_top1"],"micro_top1":v["micro_top1"]} for name,v in values.items()},
        "comparisons":comparisons,"transitions":{"corrected":int(corrected.sum()),"damaged":int(damaged.sum()),"net":int(corrected.sum()-damaged.sum())},
        "policy":{"full_histogram":[int(x) for x in histogram],"highest_occupancy":highest,"used_actions":int(histogram.gt(0).sum()),"mean_entropy":float(full_entropy.mean()),"center_overlap":float(full_actions.eq(12).double().mean()),"static_best_action":static_action,"static_best_is_center":static_action==12,"static_train_losses":static_losses,"full_soff_agreement":float(full_actions.eq(soff_actions).double().mean()),"full_image_only_agreement":float(full_actions.eq(image_actions).double().mean())},
        "oracle":{"macro_top1":100*float(oracle_vector.mean()),"all25_diagnostic_only":True},
        "checkpoint_identities":{
            "full":{"sha256":config["full_checkpoint"]["sha256"],"training_commit":config["full_checkpoint"]["training_commit"],"train_config_sha256":full_cp["config_sha256"],"gradient_receipt":full_cp["gradient_receipt"]},
            "image_only":{"sha256":config["image_only_checkpoint"]["sha256"],"training_commit":config["image_only_checkpoint"]["training_commit"],"train_config_sha256":image_cp["config_sha256"],"gradient_receipt":image_cp["gradient_receipt"]},
        },
        "b1_receipt":{
            "raw_original_open_count_by_condition":{
                "full":len(paths),"s_off":len(paths),"image_only":len(paths),
                "center":len(paths),"static_best":len(paths),"v_off":0,"i_off_reuses_full":0,
            },
            "total_control_raw_original_open_count":len(paths)*5,
            "full_selected_crop_forward_count":len(paths),
            "all25_full_eval_encoding_count":0,
            "v_off_lowres_selected_crop_forward_count":len(paths),
            "v_off_all25_eval_encoding_count":0,
            "v_off_preprocessed_336_opened":True,
            "full_boxes_sample":full_boxes[0].tolist(),
            "full_actions_sha256":__import__("hashlib").sha256(full_actions.numpy().tobytes()).hexdigest(),
            "full_selected_boxes_sha256":__import__("hashlib").sha256(full_boxes.numpy().tobytes()).hexdigest(),
            "policy_decision_before_raw_open":True,
        },
        "module_call_counts":{"full_policy":full_policy_calls,"full_update":full_update_calls,"s_off_policy":soff_policy_calls,"s_off_update":soff_update_calls,"v_off":voff_calls,"i_off":ioff_calls,"image_only_policy":image_policy_calls,"image_only_update":image_update_calls},
        "gates":gates,"preliminary_gate_passed":preliminary,"decision":"continue_remaining_controls" if preliminary else "drop_cuav_preliminary_gate",
        "unseen_images_used_for_gradient":False,"official_test_loaded":False,"pclr_online_inference":False,
    }
    output=prepare_output_dir(output_path); atomic_write_json(output/("result.json" if preliminary else "failure.json"),result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,required=True); parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--expected-commit",required=True); parser.add_argument("--expected-config-sha",required=True)
    args=parser.parse_args(); print(run(args.config,args.output,args.expected_commit,args.expected_config_sha))


if __name__=="__main__": main()
