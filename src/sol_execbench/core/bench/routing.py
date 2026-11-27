"""Correctness checks for FP8 MoE gate routing."""

import math

import torch

from sol_execbench.core.bench.correctness import check_tensor_sanity
from sol_execbench.core.data.trace import Correctness
from sol_execbench.core.data.workload import ToleranceSpec

FP8_MOE_ROUTER = "011_fp8_moe_gate_routing"


def non_saturating_inputs(inputs: dict) -> dict:
    """Reduce activation magnitude and recompute FP8 scales from the rounded BF16 input."""
    hidden = inputs["hidden_states"]
    hidden = (hidden.float() / math.sqrt(hidden.shape[-1])).to(hidden.dtype)
    block_max = hidden.unflatten(-1, (-1, 128)).abs().amax(dim=-1).float()
    scales = (block_max / torch.finfo(torch.float8_e4m3fn).max).clamp_min_(1e-12)
    return {**inputs, "hidden_states": hidden, "scale_x": scales}


def compare_routing(output, reference, tolerance: ToleranceSpec, experts: int):
    """Match each token's unordered (expert ID, weight) pairs at the workload tolerance."""
    ids, weights = output
    ref_ids, ref_weights = reference
    if any(
        actual.shape != expected.shape or actual.dtype != expected.dtype
        for actual, expected in zip(output, reference, strict=True)
    ):
        return Correctness(extra={"routing_error": "wrong shape or dtype"}), True
    if ids.numel() == 0:
        return Correctness(), False
    if ((ids < 0) | (ids >= experts)).any().item() or (
        ids.sort().values.diff() == 0
    ).any().item():
        return Correctness(
            extra={"routing_error": "out-of-range or duplicate expert"}
        ), True
    sanity = check_tensor_sanity(weights, ref_weights)
    if sanity is not None:
        return sanity, True

    same_id = ids.unsqueeze(-1) == ref_ids.unsqueeze(-2)
    eligible = same_id.any(-1)
    expected = ref_weights.gather(-1, same_id.to(torch.uint8).argmax(-1))
    expected = expected.float()
    error = (weights.float() - expected).abs().masked_fill(~eligible, 0.0)
    matched = eligible & (
        error <= tolerance.max_atol + tolerance.max_rtol * expected.abs()
    )
    ratio = matched.sum().item() / ids.numel()
    max_error = error.max().item()
    if tolerance.max_error_cap is not None:
        remaining = weights.masked_fill(same_id.any(-1), torch.inf).sort().values
        missing = ref_weights.masked_fill(same_id.any(-2), torch.inf).sort().values
        residual = (remaining - missing).masked_fill(missing.isinf(), 0.0)
        max_error = max(max_error, residual.abs().max().item())
    denominator = expected.abs().clamp_min(
        max(tolerance.max_atol, torch.finfo(torch.float32).tiny)
    )
    stats = Correctness(
        max_absolute_error=max_error,
        max_relative_error=(error / denominator).max().item(),
        extra={"matched_pair_ratio": ratio},
    )
    return stats, ratio < tolerance.required_matched_ratio or (
        tolerance.max_error_cap is not None and max_error > tolerance.max_error_cap
    )
