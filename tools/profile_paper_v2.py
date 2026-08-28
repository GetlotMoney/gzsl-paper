"""Profile frozen CLIP and the final three-module inference path on one GPU."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch.profiler import ProfilerActivity, profile

from model.candidates.v2.trainers.paper_v2 import build_three_module_model, load_assets, load_config
from tools.runtime import sha256_file


def _cuda_latencies(callable_, warmup: int, repeats: int) -> list[float]:
    for _ in range(int(warmup)):
        callable_()
    torch.cuda.synchronize()
    values = []
    for _ in range(int(repeats)):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        callable_()
        end.record()
        torch.cuda.synchronize()
        values.append(float(start.elapsed_time(end)))
    return values


def _summary(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    return {
        "mean_ms": statistics.fmean(values),
        "median_ms": statistics.median(values),
        "p95_ms": ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))],
    }


def run(config_path: Path, model_path: Path, output: Path, device_name: str, warmup: int, repeats: int) -> dict:
    if output.exists():
        raise FileExistsError(f"profile输出已存在：{output}")
    config, config_sha = load_config(config_path)
    tensors, manifest, _ = load_assets(config)
    payload = torch.load(model_path, map_location="cpu", weights_only=False)
    if payload.get("config_sha256") != config_sha:
        raise ValueError("profile模型与配置SHA不一致。")
    device = torch.device(device_name)
    model = build_three_module_model(config, tensors, manifest, device)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()
    with torch.inference_mode():
        prototype_start = time.perf_counter()
        prototypes = F.normalize(model.prototypes(), dim=-1)
        torch.cuda.synchronize()
        prototype_ms = (time.perf_counter() - prototype_start) * 1000.0
        scale = model.scale().detach()
        image = F.normalize(tensors["test_seen_features"][:1].to(device).float(), dim=-1)
        batch64 = F.normalize(tensors["test_seen_features"][:64].to(device).float(), dim=-1)

        def head_one():
            return image @ prototypes.T * scale

        def head_64():
            return batch64 @ prototypes.T * scale

        torch.cuda.reset_peak_memory_stats(device)
        one_values = _cuda_latencies(head_one, warmup, repeats)
        memory_after_one = int(torch.cuda.max_memory_allocated(device))
        batch_values = _cuda_latencies(head_64, warmup, repeats)
        memory_after_batch = int(torch.cuda.max_memory_allocated(device))

    import clip

    clip_checkpoint = manifest["source_uris"]["clip_checkpoint"]
    clip_model, preprocess = clip.load(clip_checkpoint, device=device, jit=False)
    clip_model.eval()
    raw_image = Path(manifest["raw_image_example_uri"])
    preprocess_values = []
    image_tensor = None
    for _ in range(max(10, min(100, int(repeats)))):
        start = time.perf_counter()
        with Image.open(raw_image) as opened:
            current = preprocess(opened.convert("RGB"))
        preprocess_values.append((time.perf_counter() - start) * 1000.0)
        image_tensor = current.unsqueeze(0).to(device)

    @torch.inference_mode()
    def full_one():
        encoded = F.normalize(clip_model.encode_image(image_tensor).float(), dim=-1)
        return encoded @ prototypes.T * scale

    torch.cuda.reset_peak_memory_stats(device)
    full_values = _cuda_latencies(full_one, warmup, repeats)
    full_memory = int(torch.cuda.max_memory_allocated(device))
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], with_flops=True) as profiler:
        full_one()
        torch.cuda.synchronize()
    profiled_flops = sum(int(event.flops or 0) for event in profiler.key_averages())

    active_parameters = {
        id(parameter): parameter
        for parameters in model.parameter_groups().values()
        for parameter in parameters
    }
    class_count = int(manifest["class_count"])
    head_macs_batch1 = class_count * 768
    result = {
        "schema_version": "gzsl-paper.efficiency.v1",
        "dataset": config["dataset"],
        "config_sha256": config_sha,
        "model_sha256": sha256_file(model_path),
        "device": torch.cuda.get_device_name(device),
        "warmup": int(warmup),
        "repeats": int(repeats),
        "total_head_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "active_trainable_parameters": sum(parameter.numel() for parameter in active_parameters.values()),
        "frozen_clip_parameters": sum(parameter.numel() for parameter in clip_model.parameters()),
        "clip_plus_head_parameters": sum(parameter.numel() for parameter in clip_model.parameters())
        + sum(parameter.numel() for parameter in model.parameters()),
        "prototype_precompute_ms": prototype_ms,
        "cosine_head_macs_batch1": head_macs_batch1,
        "cosine_head_flops_batch1": 2 * head_macs_batch1,
        "head_batch1_latency": _summary(one_values),
        "head_batch64_latency": _summary(batch_values),
        "head_batch64_images_per_second": 64000.0 / statistics.fmean(batch_values),
        "raw_image_preprocess_latency": _summary(preprocess_values),
        "clip_plus_head_batch1_latency": _summary(full_values),
        "clip_plus_head_profiled_flops_batch1": profiled_flops,
        "peak_memory_batch1_bytes": memory_after_one,
        "peak_memory_batch64_bytes": memory_after_batch,
        "peak_memory_clip_plus_head_batch1_bytes": full_memory,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=500)
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.model, args.output, args.device, args.warmup, args.repeats), ensure_ascii=False))


if __name__ == "__main__":
    main()
