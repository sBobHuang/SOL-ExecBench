import pytest
import torch

from sol_execbench.core.bench.routing import compare_routing, non_saturating_inputs
from sol_execbench.core.data.workload import ToleranceSpec


def pairs(ids, weights):
    return torch.tensor([ids]), torch.tensor([weights], dtype=torch.float32)


def test_pair_permutation_and_exact_ids():
    ref = pairs([150, 200, 206], [0.2, 0.3, 0.5])
    tol = ToleranceSpec(max_atol=0, max_rtol=0.02, required_matched_ratio=1)
    assert not compare_routing(pairs([206, 150, 200], [0.5, 0.2, 0.3]), ref, tol, 256)[
        1
    ]
    assert compare_routing(pairs([206, 150, 200], [0.2, 0.3, 0.5]), ref, tol, 256)[1]
    stats, failed = compare_routing(
        pairs([206, 150, 201], [0.5, 0.2, 0.3]), ref, tol, 256
    )
    assert failed and stats.extra["matched_pair_ratio"] == pytest.approx(2 / 3)
    assert not compare_routing(
        pairs([206, 150, 201], [0.5, 0.2, 0.3]),
        ref,
        tol.model_copy(update={"required_matched_ratio": 2 / 3}),
        256,
    )[1]


@pytest.mark.parametrize("ids", [[1, 1], [-1, 1], [2, 10]])
def test_invalid_ids_are_not_tolerated(ids):
    assert compare_routing(
        pairs(ids, [0.5, 0.5]),
        pairs([1, 2], [0.5, 0.5]),
        ToleranceSpec(required_matched_ratio=0),
        10,
    )[1]


def test_weight_sanity_and_cap():
    ref = pairs([1, 2], [0.3, 0.7])
    tol = ToleranceSpec(max_atol=0.5, required_matched_ratio=0.5, max_error_cap=0.05)
    for value in [float("nan"), float("inf"), 0.8]:
        assert compare_routing(pairs([2, 1], [value, 0.3]), ref, tol, 3)[1]
    assert compare_routing((ref[0].float(), ref[1]), ref, tol, 3)[1]
    assert compare_routing((ref[0][:, :1], ref[1]), ref, tol, 3)[1]
    assert compare_routing(pairs([0, 1], [10.0, 0.3]), ref, tol, 3)[1]
    empty = ref[0][:0], ref[1][:0]
    assert not compare_routing(empty, empty, tol, 3)[1]


def test_fp8_router_case_exposes_skipped_low_bias_expert():
    hidden = torch.ones(1, 7168, dtype=torch.bfloat16)
    inputs = {"hidden_states": hidden, "scale_x": torch.full((1, 56), 1 / 448)}
    tolerance = ToleranceSpec(
        max_atol=0.005, max_rtol=0.02, required_matched_ratio=0.99
    )
    low_bias_expert = 4 * 32 + 31
    projection = torch.full((256, 7168), 1 / 64)
    projection[low_bias_expert].fill_(1 / 32)
    bias = torch.full((8, 32), -0.25, dtype=torch.bfloat16)
    bias[0, :7], bias[0, 7] = 1 / 32, 0
    for group in range(1, 5):
        bias[group] = -(group + 1) / 64
    bias[4, -1] = -6 / 64
    expert_ids = torch.arange(256).view(8, 32)
    high_bias_output = pairs(list(range(8)), [2.5 / 8] * 8)

    for alternate in (False, True):
        if alternate:
            inputs = non_saturating_inputs(inputs)
        logits = (inputs["hidden_states"].float() @ projection.T).bfloat16()
        scores = logits.float().sigmoid()
        choices = scores.view(8, 32) + bias
        group_scores, group_ids = choices.topk(2).values.sum(-1).sort(descending=True)
        assert (group_scores[3] - group_scores[4]).item() >= 0.03
        groups = group_ids[:4]
        assert (4 in groups.tolist()) == alternate
        candidates = expert_ids[groups].flatten()
        top_scores, slots = choices.flatten()[candidates].topk(9)
        assert (top_scores[7] - top_scores[8]).item() >= 0.03
        ids = candidates[slots[:8]].view(1, 8)
        expected_last = low_bias_expert if alternate else 7
        assert set(ids[0].tolist()) == {*range(7), expected_last}
        weights = scores.gather(1, ids)
        reference = ids, weights / weights.sum(-1, keepdim=True) * 2.5
        stats, failed = compare_routing(high_bias_output, reference, tolerance, 256)
        assert stats.extra["matched_pair_ratio"] == (7 / 8 if alternate else 1)
        assert failed == alternate


def test_pairs_cannot_move_between_tokens():
    ref_ids = torch.tensor([[0, 1], [2, 3]])
    weights = torch.full_like(ref_ids, 0.5, dtype=torch.float32)
    stats, failed = compare_routing(
        (ref_ids.flip((0,)), weights), (ref_ids, weights), ToleranceSpec(), 4
    )
    assert failed and stats.extra["matched_pair_ratio"] == 0
