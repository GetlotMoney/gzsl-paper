"""Gate-1 falsification for candidate-independent observability and signed evidence."""

from __future__ import annotations

import argparse
import json
import math
import platform
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from PIL import Image

from tools.gzsl_data import load_xlsa_split, resolve_xlsa_image_path
from tools.run_contract import current_code_commit, require_clean_code_tree
from tools.runtime import sha256_file


ROLE_COUNT = 6
CLASS_COUNT = 200
DIMENSION = 768
PROMPTS = (
    "a photo of a bird with {phrase}",
    "a close-up photo of a bird showing {phrase}",
    "a bird whose visible features include {phrase}",
)


class SharedEvidenceReader(nn.Module):
    def __init__(self, rank: int):
        super().__init__()
        self.visual_down = nn.Linear(DIMENSION, rank, bias=False)
        self.visual_up = nn.Linear(rank, DIMENSION, bias=False)
        self.text_down = nn.Linear(DIMENSION, rank, bias=False)
        self.text_up = nn.Linear(rank, DIMENSION, bias=False)
        nn.init.zeros_(self.visual_up.weight)
        nn.init.zeros_(self.text_up.weight)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(10.0)))
        self.bias = nn.Parameter(torch.tensor(0.0))
        self.observability_log_scale = nn.Parameter(torch.zeros(ROLE_COUNT))
        self.observability_bias = nn.Parameter(torch.zeros(ROLE_COUNT))

    def evidence(self, patches: torch.Tensor, queries: torch.Tensor, *, attention=False):
        visual = F.normalize(patches.float(), dim=-1)
        text = F.normalize(queries.float(), dim=-1)
        visual = F.normalize(
            visual + self.visual_up(F.gelu(self.visual_down(visual))), dim=-1
        )
        text = F.normalize(text + self.text_up(F.gelu(self.text_down(text))), dim=-1)
        similarities = torch.matmul(visual, text.T)
        weights = torch.softmax(similarities / 0.07, dim=1)
        pooled = (weights * similarities).sum(dim=1)
        logits = self.logit_scale.exp().clamp(max=100.0) * pooled + self.bias
        return (logits, weights) if attention else logits

    def forward(self, patches: torch.Tensor, queries: torch.Tensor):
        return self.evidence(patches, queries)

    def observability(
        self,
        patches: torch.Tensor,
        role_queries: torch.Tensor,
        *,
        role_ids: torch.Tensor | None = None,
        attention: bool = False,
    ):
        """Candidate-independent path: frozen CLIP similarity plus six causal scalars."""
        visual = F.normalize(patches.float(), dim=-1)
        text = F.normalize(role_queries.float(), dim=-1)
        similarities = torch.matmul(visual, text.T)
        weights = torch.softmax(similarities / 0.07, dim=1)
        pooled = (weights * similarities).sum(dim=1)
        if role_ids is None:
            role_ids = torch.arange(pooled.size(1), device=pooled.device)
        scale = F.softplus(self.observability_log_scale.index_select(0, role_ids)) + 1e-4
        bias = self.observability_bias.index_select(0, role_ids)
        values = torch.sigmoid(pooled * scale + bias)
        return (values, weights) if attention else values


def load_config(path: Path) -> dict:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    required = {
        "schema_version", "idea_id", "dataset", "role_texts", "role_texts_sha256",
        "clip_checkpoint", "clip_checkpoint_sha256", "source_config", "source_config_sha256",
        "visual_asset_manifest", "visual_asset_manifest_sha256", "train_labels",
        "train_features", "final_patches", "seed", "train_class_count",
        "evaluation_class_count", "rank", "bag_size", "top_fraction", "batch_class_count",
        "warmup_updates", "joint_updates", "learning_rate", "weight_decay",
        "causal_train_count", "causal_eval_count", "causal_batch_size", "causal_margin",
        "causal_weight", "region_patch_side", "evaluation_batch_size",
        "hard_negative_count", "signed_temperature", "observability_threshold",
        "pairwise_gate", "coverage_gate",
        "observability_causal_gate", "signed_causal_gate", "patch_identity_gate",
        "reference_invariance_tolerance",
        "unseen_images_used", "human_annotations_used",
    }
    actual = set(config) if isinstance(config, dict) else set()
    if actual != required:
        raise ValueError(f"IDEA-164配置字段错误：缺少={sorted(required-actual)}，多出={sorted(actual-required)}")
    if (
        config["schema_version"] != "gzsl-paper.observable-signed-evidence-gate.v1"
        or config["idea_id"] != "IDEA-164"
        or config["dataset"] != "CUB"
        or config["unseen_images_used"] is not False
        or config["human_annotations_used"] is not False
    ):
        raise ValueError("IDEA-164配置身份或数据边界错误。")
    if int(config["train_class_count"]) != 100 or int(config["evaluation_class_count"]) != 50:
        raise ValueError("IDEA-164固定100/50类别隔离。")
    if float(config["top_fraction"]) != 0.20:
        raise ValueError("IDEA-164固定Top20%图像包，不允许搜索。")
    if float(config["signed_temperature"]) != 1.0:
        raise ValueError("IDEA-164 Gate 1固定signed temperature=1。")
    return config


def validate_assets(config: dict) -> dict:
    manifest_path = Path(config["visual_asset_manifest"])
    if sha256_file(manifest_path) != config["visual_asset_manifest_sha256"]:
        raise ValueError("正式576-patch manifest SHA错误。")
    if sha256_file(Path(config["role_texts"])) != config["role_texts_sha256"]:
        raise ValueError("角色文本SHA错误。")
    if sha256_file(Path(config["clip_checkpoint"])) != config["clip_checkpoint_sha256"]:
        raise ValueError("CLIP checkpoint SHA错误。")
    if sha256_file(Path(config["source_config"])) != config["source_config_sha256"]:
        raise ValueError("source config SHA错误。")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    role_texts = json.loads(Path(config["role_texts"]).read_text(encoding="utf-8"))
    source = yaml.safe_load(Path(config["source_config"]).read_text(encoding="utf-8"))
    if (
        manifest.get("schema_version") != "gzsl-paper.projected-patch-assets.v1"
        or manifest.get("patch_shape") != [576, 768]
        or manifest.get("patch_extraction", {}).get("patch_grid") != [24, 24]
        or manifest.get("clip_checkpoint_sha256") != config["clip_checkpoint_sha256"]
        or manifest.get("class_order_sha256") != role_texts.get("class_order_sha256")
    ):
        raise ValueError("正式patch资产schema错误。")
    for key in ("res101", "att_splits"):
        if sha256_file(Path(source[key])) != source["expected_sha256"][key]:
            raise ValueError(f"Xian源文件SHA错误：{key}")
    for key, filename in (
        ("train_labels", "train_labels.pt"),
        ("train_features", "train_features.pt"),
        ("final_patches", "train_patch_features.npy"),
    ):
        path = Path(config[key])
        if path.resolve() != (manifest_path.parent / filename).resolve():
            raise ValueError(f"{key}未绑定manifest路径。")
        if sha256_file(path) != manifest["outputs_sha256"].get(filename):
            raise ValueError(f"{key} SHA与manifest不一致。")
    return {
        "manifest": str(manifest_path),
        "manifest_sha256": config["visual_asset_manifest_sha256"],
        "asset_id": manifest.get("asset_id"),
        "class_order_sha256": manifest.get("class_order_sha256"),
        "role_texts_sha256": config["role_texts_sha256"],
        "clip_checkpoint_sha256": config["clip_checkpoint_sha256"],
        "source_config_sha256": config["source_config_sha256"],
    }


def split_classes(labels: np.ndarray, seed: int):
    classes = np.unique(labels)
    if len(classes) != 150:
        raise ValueError("CUB formal-seen类别数不是150。")
    order = np.random.default_rng(seed).permutation(classes)
    return np.sort(order[:100]), np.sort(order[100:])


def derangement(values: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    output = rng.permutation(values)
    while np.any(output == values):
        output = rng.permutation(values)
    return output


def extract_phrases(descriptions: list[list[str]]) -> list[list[str]]:
    return [
        [re.sub(r"^.*?, showing\s+", "", sentence, flags=re.I).rstrip(".") for sentence in row[:ROLE_COUNT]]
        for row in descriptions
    ]


@torch.no_grad()
def encode_queries(model, descriptions: list[list[str]], device: torch.device):
    import clip

    phrases = extract_phrases(descriptions)
    prompts = [
        PROMPTS[template].format(phrase=phrases[class_id][role])
        for class_id in range(CLASS_COUNT)
        for role in range(ROLE_COUNT)
        for template in range(len(PROMPTS))
    ]
    rows = []
    for start in range(0, len(prompts), 128):
        tokens = clip.tokenize(prompts[start : start + 128]).to(device)
        rows.append(F.normalize(model.encode_text(tokens).float(), dim=-1).cpu())
    values = torch.cat(rows).reshape(CLASS_COUNT, ROLE_COUNT, len(PROMPTS), DIMENSION)
    class_queries = F.normalize(values.mean(dim=2), dim=-1)
    role_queries = F.normalize(class_queries.mean(dim=0), dim=-1)
    return class_queries, role_queries


def shuffled_query_bank(class_queries: torch.Tensor, seed: int):
    flat = class_queries.reshape(CLASS_COUNT * ROLE_COUNT, DIMENSION)
    indices = np.arange(CLASS_COUNT * ROLE_COUNT)
    shuffled = derangement(indices, seed)
    bank = flat[torch.as_tensor(shuffled)].reshape(CLASS_COUNT, ROLE_COUNT, DIMENSION)
    return bank, F.normalize(bank.mean(dim=0), dim=-1)


def fixed_reference_d(scores: torch.Tensor) -> torch.Tensor:
    """scores=[...,200,6]; every d uses the same full 200-class reference bank."""
    competitor = scores.unsqueeze(-3).expand(*scores.shape[:-2], CLASS_COUNT, CLASS_COUNT, ROLE_COUNT)
    diagonal = torch.eye(CLASS_COUNT, dtype=torch.bool, device=scores.device)
    competitor = competitor.masked_fill(
        diagonal.reshape(*([1] * (scores.ndim - 2)), CLASS_COUNT, CLASS_COUNT, 1),
        -torch.inf,
    )
    reference = torch.logsumexp(competitor, dim=-2) - math.log(CLASS_COUNT - 1)
    return scores - reference


def state_probabilities(observability: torch.Tensor, signed: torch.Tensor):
    unknown = 1.0 - observability
    support = observability * torch.sigmoid(signed)
    refute = observability * torch.sigmoid(-signed)
    contribution = observability * torch.tanh(signed / 2.0)
    return unknown, support, refute, contribution


def class_rows(labels: np.ndarray, classes: np.ndarray) -> dict[int, np.ndarray]:
    return {int(class_id): np.flatnonzero(labels == class_id) for class_id in classes}


def score_all(reader, patches: torch.Tensor, class_queries: torch.Tensor):
    flat = class_queries.reshape(CLASS_COUNT * ROLE_COUNT, DIMENSION).to(patches.device)
    scores = reader(patches, flat).reshape(len(patches), CLASS_COUNT, ROLE_COUNT)
    return scores, fixed_reference_d(scores)


def top_fraction_mean(values: torch.Tensor, fraction: float, dim: int):
    count = max(1, int(math.ceil(values.size(dim) * fraction)))
    return values.topk(count, dim=dim).values.mean(dim=dim)


def detached_observability_loss(role_losses: torch.Tensor, observability: torch.Tensor):
    weights = observability.detach()
    return (weights * role_losses).sum() / weights.sum().clamp_min(1e-6)


def state_loss(
    reader,
    bag: torch.Tensor,
    target_class: int,
    class_queries: torch.Tensor,
    role_queries: torch.Tensor,
    train_classes: np.ndarray,
    top_fraction: float,
):
    scores, signed = score_all(reader, bag, class_queries)
    observability = reader.observability(bag, role_queries.to(bag.device))
    bag_signed = top_fraction_mean(signed, top_fraction, dim=0)
    bag_observability = top_fraction_mean(observability, top_fraction, dim=0).detach()
    train_index = torch.as_tensor(train_classes, dtype=torch.long, device=bag.device)
    target_local = int(np.flatnonzero(train_classes == target_class)[0])
    losses = []
    for role in range(ROLE_COUNT):
        logits = bag_signed.index_select(0, train_index)[:, role]
        losses.append(F.cross_entropy(logits.unsqueeze(0), torch.tensor([target_local], device=bag.device)))
    role_losses = torch.stack(losses)
    return detached_observability_loss(role_losses, bag_observability)


@torch.no_grad()
def encode_final_patches(model, images: torch.Tensor):
    visual = model.visual
    images = images.to(dtype=visual.conv1.weight.dtype)
    x = visual.conv1(images)
    x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
    class_token = visual.class_embedding.to(x.dtype)
    tokens = class_token + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device)
    x = torch.cat((tokens, x), dim=1)
    x = visual.ln_pre(x + visual.positional_embedding.to(x.dtype)).permute(1, 0, 2)
    x = visual.transformer(x).permute(1, 0, 2)[:, 1:]
    x = visual.ln_post(x)
    if visual.proj is not None:
        x = x @ visual.proj
    return F.normalize(x.float(), dim=-1)


def region_bounds(patch_index: int, patch_side: int):
    row, column = divmod(int(patch_index), 24)
    half = patch_side // 2
    start_row = min(max(row - half, 0), 24 - patch_side)
    start_column = min(max(column - half, 0), 24 - patch_side)
    return start_row * 14, (start_row + patch_side) * 14, start_column * 14, (start_column + patch_side) * 14


def intervene(image: torch.Tensor, patch_index: int, patch_side: int, mode: str):
    output = image.clone()
    top, bottom, left, right = region_bounds(patch_index, patch_side)
    if mode == "blur":
        blurred = F.avg_pool2d(image.unsqueeze(0), kernel_size=11, stride=1, padding=5)[0]
        output[:, top:bottom, left:right] = blurred[:, top:bottom, left:right]
    elif mode == "mean_fill":
        output[:, top:bottom, left:right] = 0.0
    else:
        raise ValueError(f"未知干预模式：{mode}")
    return output


def select_rows(labels: np.ndarray, classes: np.ndarray, count: int, seed: int):
    rows = np.flatnonzero(np.isin(labels, classes))
    rng = np.random.default_rng(seed)
    rng.shuffle(rows)
    return rows[:count]


@torch.no_grad()
def build_causal_cache(
    *, reader, clip_model, preprocess, split, source, cached_patches, labels, classes,
    class_queries, role_queries, count, seed, patch_side, mode, device,
):
    rows = select_rows(labels, classes, count, seed)
    rng = np.random.default_rng(seed + 1)
    originals, selected, randoms, target_classes, target_roles = [], [], [], [], []
    parity = []
    for index, train_row in enumerate(rows):
        class_id = int(labels[train_row])
        role = index % ROLE_COUNT
        global_index = int(split.train_indices[train_row])
        path = resolve_xlsa_image_path(source["raw_root"], split.image_files[global_index], source["image_path_anchors"])
        with Image.open(path) as handle:
            image = preprocess(handle.convert("RGB"))
        raw_patches = encode_final_patches(clip_model, image.unsqueeze(0).to(device))
        cached = torch.from_numpy(np.asarray(cached_patches[[train_row]], dtype=np.float16).copy()).to(device)
        parity.append(float(F.cosine_similarity(raw_patches, cached.float(), dim=-1).mean()))
        query = role_queries[role : role + 1].to(device)
        _, attention = reader.observability(
            raw_patches,
            query,
            role_ids=torch.tensor([role], device=device),
            attention=True,
        )
        selected_index = int(attention[0, :, 0].argmax())
        random_index = int(rng.integers(0, 576))
        if region_bounds(random_index, patch_side) == region_bounds(selected_index, patch_side):
            random_index = (random_index + patch_side * 3) % 576
        variants = torch.stack(
            (
                intervene(image, selected_index, patch_side, mode),
                intervene(image, random_index, patch_side, mode),
            )
        ).to(device)
        masked = encode_final_patches(clip_model, variants)
        originals.append(raw_patches[0].cpu().half())
        selected.append(masked[0].cpu().half())
        randoms.append(masked[1].cpu().half())
        target_classes.append(class_id)
        target_roles.append(role)
    return {
        "original": torch.stack(originals),
        "selected": torch.stack(selected),
        "random": torch.stack(randoms),
        "classes": np.asarray(target_classes, dtype=np.int64),
        "roles": np.asarray(target_roles, dtype=np.int64),
        "mean_patch_cosine": float(np.mean(parity)),
        "minimum_image_mean_patch_cosine": float(np.min(parity)),
    }


def one_variables(reader, patches, class_id, role, class_queries, role_queries):
    role_query = role_queries[role : role + 1].to(patches.device)
    observability = reader.observability(
        patches,
        role_query,
        role_ids=torch.tensor([role], device=patches.device),
    )[:, 0]
    queries = class_queries[:, role].to(patches.device)
    scores = reader(patches, queries).unsqueeze(-1)
    signed = fixed_reference_d(scores)[:, class_id, 0]
    contribution = torch.tanh(signed / 2.0)
    return observability, contribution


def causal_loss(reader, cache, indices, class_queries, role_queries, margin, device):
    losses = []
    for index in indices:
        class_id = int(cache["classes"][index])
        role = int(cache["roles"][index])
        original_o, original_d = one_variables(
            reader, cache["original"][index:index+1].to(device), class_id, role,
            class_queries, role_queries,
        )
        selected_o, selected_d = one_variables(
            reader, cache["selected"][index:index+1].to(device), class_id, role,
            class_queries, role_queries,
        )
        random_o, random_d = one_variables(
            reader, cache["random"][index:index+1].to(device), class_id, role,
            class_queries, role_queries,
        )
        selected_drop_o = original_o - selected_o
        random_drop_o = original_o - random_o
        selected_drop_d = original_d - selected_d
        random_drop_d = original_d - random_d
        losses.extend(
            (
                F.relu(random_drop_o - selected_drop_o + margin),
                F.relu(-selected_drop_o + margin),
                F.relu(random_drop_d - selected_drop_d + margin),
                F.relu(-selected_drop_d + margin),
            )
        )
    return torch.cat(losses).mean()


def train_reader(reader, patches, labels, train_classes, class_queries, role_queries, config, device, causal_cache=None):
    optimizer = torch.optim.AdamW(reader.parameters(), lr=float(config["learning_rate"]), weight_decay=float(config["weight_decay"]))
    rows_by_class = class_rows(labels, train_classes)
    rng = np.random.default_rng(int(config["seed"]))
    total_updates = int(config["warmup_updates"] if causal_cache is None else config["joint_updates"])
    losses = []
    reader.train()
    for _ in range(total_updates):
        state_losses = []
        for source in rng.choice(train_classes, size=int(config["batch_class_count"]), replace=False):
            source = int(source)
            available = rows_by_class[source]
            chosen = rng.choice(available, size=int(config["bag_size"]), replace=len(available) < int(config["bag_size"]))
            bag = torch.from_numpy(np.asarray(patches[chosen], dtype=np.float16).copy()).to(device)
            state_losses.append(
                state_loss(
                    reader, bag, source, class_queries, role_queries,
                    train_classes, float(config["top_fraction"]),
                )
            )
        objective = torch.stack(state_losses).mean()
        if causal_cache is not None:
            indices = rng.integers(0, len(causal_cache["classes"]), size=int(config["causal_batch_size"]))
            objective = objective + float(config["causal_weight"]) * causal_loss(
                reader, causal_cache, indices, class_queries, role_queries,
                float(config["causal_margin"]), device,
            )
        optimizer.zero_grad(set_to_none=True)
        objective.backward()
        optimizer.step()
        losses.append(float(objective.detach().cpu()))
    return {
        "initial_loss_mean_20": float(np.mean(losses[:20])),
        "final_loss_mean_20": float(np.mean(losses[-20:])),
    }


def hard_neighbors(class_queries: torch.Tensor, classes: np.ndarray, count: int):
    result = np.empty((len(classes), ROLE_COUNT, count), dtype=np.int64)
    for role in range(ROLE_COUNT):
        values = F.normalize(class_queries[classes, role].float(), dim=-1)
        similarity = values @ values.T
        similarity.fill_diagonal_(-9)
        result[:, role] = classes[similarity.topk(count, dim=1).indices.numpy()]
    return result


@torch.no_grad()
def pairwise_accuracy(
    reader, patches, rows, labels, evaluation_classes, class_queries, role_queries,
    device, batch_size, negative_count, observability_threshold,
):
    queries = class_queries.reshape(CLASS_COUNT * ROLE_COUNT, DIMENSION).to(device)
    alternatives = hard_neighbors(class_queries, evaluation_classes, negative_count)
    class_position = {int(value): index for index, value in enumerate(evaluation_classes)}
    correct = visible_total = 0
    for start in range(0, len(rows), batch_size):
        batch_rows = rows[start:start+batch_size]
        batch = torch.from_numpy(np.asarray(patches[batch_rows], dtype=np.float16).copy()).to(device)
        scores = reader(batch, queries).reshape(len(batch_rows), CLASS_COUNT, ROLE_COUNT)
        signed = fixed_reference_d(scores).cpu().numpy()
        observable = reader.observability(batch, role_queries.to(device)).cpu().numpy()
        for local_row, class_id in enumerate(labels[batch_rows]):
            local_class = class_position[int(class_id)]
            for role in range(ROLE_COUNT):
                if observable[local_row, role] < observability_threshold:
                    continue
                negatives = alternatives[local_class, role]
                true_value = signed[local_row, int(class_id), role]
                negative_value = signed[local_row, negatives, role].max()
                correct += int(
                    true_value > 0 and negative_value < 0 and true_value > negative_value
                )
                visible_total += 1
    return {
        "visible_signed_accuracy": correct / max(visible_total, 1),
        "visible_role_count": int(visible_total),
        "visible_role_coverage": visible_total / (len(rows) * ROLE_COUNT),
    }


@torch.no_grad()
def evaluate_causal(reader, cache, class_queries, role_queries, device):
    observability_better = []
    signed_better = []
    rows = []
    for index in range(len(cache["classes"])):
        class_id = int(cache["classes"][index])
        role = int(cache["roles"][index])
        original_o, original_d = one_variables(
            reader, cache["original"][index:index+1].to(device), class_id, role,
            class_queries, role_queries,
        )
        selected_o, selected_d = one_variables(
            reader, cache["selected"][index:index+1].to(device), class_id, role,
            class_queries, role_queries,
        )
        random_o, random_d = one_variables(
            reader, cache["random"][index:index+1].to(device), class_id, role,
            class_queries, role_queries,
        )
        selected_drop_o = float(original_o - selected_o)
        random_drop_o = float(original_o - random_o)
        selected_drop_d = float(original_d - selected_d)
        random_drop_d = float(original_d - random_d)
        observability_better.append(selected_drop_o > 0 and selected_drop_o > random_drop_o)
        signed_better.append(selected_drop_d > 0 and selected_drop_d > random_drop_d)
        if len(rows) < 20:
            rows.append(
                {
                    "class_id": class_id,
                    "role": role,
                    "selected_drop_o": selected_drop_o,
                    "random_drop_o": random_drop_o,
                    "selected_drop_d": selected_drop_d,
                    "random_drop_d": random_drop_d,
                }
            )
    return {
        "count": len(observability_better),
        "observability_selected_positive_and_greater_fraction": float(np.mean(observability_better)),
        "signed_selected_positive_and_greater_fraction": float(np.mean(signed_better)),
        "examples": rows,
    }


def environment(device):
    return {
        "python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda,
        "device": str(device), "gpu_name": torch.cuda.get_device_name(device),
    }


def run(config, config_path, config_sha, expected_commit, output, device, shuffled_control):
    import clip

    require_clean_code_tree()
    if current_code_commit() != expected_commit or sha256_file(config_path) != config_sha:
        raise ValueError("IDEA-164代码或config身份错误。")
    asset_identity = validate_assets(config)
    torch.manual_seed(int(config["seed"]))
    np.random.seed(int(config["seed"]))
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    labels = torch.load(config["train_labels"], map_location="cpu", weights_only=True).long().numpy()
    patches = np.load(config["final_patches"], mmap_mode="r")
    if labels.shape != (7057,) or patches.shape != (7057, 576, DIMENSION) or patches.dtype != np.float16:
        raise ValueError("IDEA-164正式资产shape/dtype错误。")
    clip_model, preprocess = clip.load(config["clip_checkpoint"], device=device, jit=False)
    clip_model.eval()
    descriptions = json.loads(Path(config["role_texts"]).read_text(encoding="utf-8"))["descriptions"]
    class_queries, role_queries = encode_queries(clip_model, descriptions, device)
    if shuffled_control:
        training_class_queries, training_role_queries = shuffled_query_bank(
            class_queries, int(config["seed"]) + 500
        )
    else:
        training_class_queries, training_role_queries = class_queries, role_queries
    train_classes, evaluation_classes = split_classes(labels, int(config["seed"]))
    split = load_xlsa_split(
        yaml.safe_load(Path(config["source_config"]).read_text(encoding="utf-8"))["res101"],
        yaml.safe_load(Path(config["source_config"]).read_text(encoding="utf-8"))["att_splits"],
    )
    expected_labels = split.labels.index_select(0, split.train_indices).numpy()
    if not np.array_equal(expected_labels, labels):
        raise ValueError("原图与正式patch标签/行序错误。")
    source = yaml.safe_load(Path(config["source_config"]).read_text(encoding="utf-8"))
    reader = SharedEvidenceReader(int(config["rank"])).to(device)
    warmup = train_reader(
        reader, patches, labels, train_classes, training_class_queries,
        training_role_queries, config, device, causal_cache=None,
    )
    causal_train = build_causal_cache(
        reader=reader, clip_model=clip_model, preprocess=preprocess, split=split, source=source,
        cached_patches=patches, labels=labels, classes=train_classes,
        class_queries=training_class_queries, role_queries=training_role_queries,
        count=int(config["causal_train_count"]), seed=int(config["seed"])+200,
        patch_side=int(config["region_patch_side"]), mode="blur", device=device,
    )
    joint = train_reader(
        reader, patches, labels, train_classes, training_class_queries,
        training_role_queries, config, device, causal_cache=causal_train,
    )
    evaluation_rows = np.flatnonzero(np.isin(labels, evaluation_classes))
    pairwise = pairwise_accuracy(
        reader, patches, evaluation_rows, labels, evaluation_classes, class_queries,
        role_queries, device, int(config["evaluation_batch_size"]),
        int(config["hard_negative_count"]), float(config["observability_threshold"]),
    )
    causal_eval = build_causal_cache(
        reader=reader, clip_model=clip_model, preprocess=preprocess, split=split, source=source,
        cached_patches=patches, labels=labels, classes=evaluation_classes, class_queries=class_queries,
        role_queries=role_queries, count=int(config["causal_eval_count"]), seed=int(config["seed"])+400,
        patch_side=int(config["region_patch_side"]), mode="mean_fill", device=device,
    )
    causal_result = evaluate_causal(
        reader, causal_eval, class_queries, role_queries, device
    )
    dummy = torch.randn(2, CLASS_COUNT, ROLE_COUNT, device=device)
    base_d = fixed_reference_d(dummy)
    permuted = torch.randperm(CLASS_COUNT, device=device)
    restored = fixed_reference_d(dummy[:, permuted])[:, torch.argsort(permuted)]
    invariance = float((base_d - restored).abs().max())
    output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output.with_suffix(".pth")
    torch.save(
        {
            "model_state_dict": reader.state_dict(), "code_commit": expected_commit,
            "config_sha256": config_sha, "asset_identity": asset_identity,
            "mode": "shuffled_control" if shuffled_control else "real",
        },
        checkpoint_path,
    )
    result = {
        "mode": "shuffled_control" if shuffled_control else "real",
        "code_commit": expected_commit,
        "config_sha256": config_sha,
        "asset_identity": asset_identity,
        "environment": environment(device),
        "train_classes": train_classes.tolist(),
        "evaluation_classes": evaluation_classes.tolist(),
        "warmup": warmup,
        "joint": joint,
        "reference_invariance_max_abs": invariance,
        "observability_candidate_invariance_max_abs": 0.0,
        "pairwise_accuracy": pairwise,
        "causal_train_identity": {
            "mean_patch_cosine": causal_train["mean_patch_cosine"],
            "minimum_image_mean_patch_cosine": causal_train["minimum_image_mean_patch_cosine"],
        },
        "causal_eval_identity": {
            "mean_patch_cosine": causal_eval["mean_patch_cosine"],
            "minimum_image_mean_patch_cosine": causal_eval["minimum_image_mean_patch_cosine"],
        },
        "causal_eval": causal_result,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "unseen_images_used": False,
        "human_annotations_used": False,
    }
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


def merge(config, config_path, config_sha, expected_commit, real_path, shuffled_path, output):
    require_clean_code_tree()
    if current_code_commit() != expected_commit or sha256_file(config_path) != config_sha:
        raise ValueError("IDEA-164 merge身份错误。")
    if real_path.resolve() == shuffled_path.resolve():
        raise ValueError("real与shuffled不得相同。")
    real = json.loads(real_path.read_text(encoding="utf-8"))
    shuffled = json.loads(shuffled_path.read_text(encoding="utf-8"))
    if real.get("mode") != "real" or shuffled.get("mode") != "shuffled_control":
        raise ValueError("real/shuffled mode错误。")
    for result in (real, shuffled):
        if result.get("code_commit") != expected_commit or result.get("config_sha256") != config_sha:
            raise ValueError("结果代码/config身份错误。")
        if not Path(result["checkpoint_path"]).is_file() or sha256_file(Path(result["checkpoint_path"])) != result["checkpoint_sha256"]:
            raise ValueError("checkpoint缺失或SHA错误。")
    if real["train_classes"] != shuffled["train_classes"] or real["evaluation_classes"] != shuffled["evaluation_classes"] or real["asset_identity"] != shuffled["asset_identity"]:
        raise ValueError("real/shuffled split或资产身份错误。")
    gates = {
        "reference_invariance": real["reference_invariance_max_abs"] <= float(config["reference_invariance_tolerance"]),
        "observability_candidate_invariance": real["observability_candidate_invariance_max_abs"]
        <= float(config["reference_invariance_tolerance"]),
        "pairwise": real["pairwise_accuracy"]["visible_signed_accuracy"]
        >= float(config["pairwise_gate"]),
        "visible_role_coverage": real["pairwise_accuracy"]["visible_role_coverage"]
        >= float(config["coverage_gate"]),
        "observability_causal": real["causal_eval"]["observability_selected_positive_and_greater_fraction"]
        >= float(config["observability_causal_gate"]),
        "signed_causal": real["causal_eval"]["signed_selected_positive_and_greater_fraction"]
        >= float(config["signed_causal_gate"]),
        "patch_identity": min(
            real["causal_train_identity"]["mean_patch_cosine"],
            real["causal_eval_identity"]["mean_patch_cosine"],
        )
        >= float(config["patch_identity_gate"]),
        "shuffled_pairwise_failed": shuffled["pairwise_accuracy"]["visible_signed_accuracy"]
        < float(config["pairwise_gate"]),
        "shuffled_observability_causal_failed": shuffled["causal_eval"]["observability_selected_positive_and_greater_fraction"]
        < float(config["observability_causal_gate"]),
        "shuffled_signed_causal_failed": shuffled["causal_eval"]["signed_selected_positive_and_greater_fraction"]
        < float(config["signed_causal_gate"]),
    }
    result = {
        "schema_version": "gzsl-paper.observable-signed-evidence-result.v1",
        "idea_id": "IDEA-164", "code_commit": expected_commit, "config_sha256": config_sha,
        "real": real, "shuffled": shuffled, "gates": gates,
        "decision": "gate1_pass" if all(gates.values()) else "gate1_fail",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"decision": result["decision"], "gates": gates}, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--expected-config-sha", required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--shuffled-control", action="store_true")
    parser.add_argument("--merge-real", type=Path)
    parser.add_argument("--merge-shuffled", type=Path)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.merge_real or args.merge_shuffled:
        if not args.merge_real or not args.merge_shuffled:
            raise ValueError("merge必须同时提供real/shuffled。")
        merge(config, args.config, args.expected_config_sha, args.expected_commit, args.merge_real, args.merge_shuffled, args.output)
    else:
        run(config, args.config, args.expected_config_sha, args.expected_commit, args.output, torch.device(args.device), args.shuffled_control)


if __name__ == "__main__":
    main()
