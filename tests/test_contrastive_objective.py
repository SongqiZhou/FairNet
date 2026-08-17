"""Regression tests for the Stage 4 contrastive objective.

Three properties are pinned here because breaking any one of them silently
turns FairNet's correction into a no-op or into an anti-correction that scores
*below* the ERM baseline it is meant to improve.
"""

import pytest
import torch
from transformers import ViTConfig

from fairnet import AttributeMode, FairNetConfig, FairNetTrainer, FairNetViT
from fairnet.modules import TripletContrastiveLoss


def _prototypical_features(scale: float):
    """A triplet with the positive nearer than the negative, at a given scale.

    The directions are fixed so that ``d_pos < d_neg`` strictly, which is the
    situation the hinge is supposed to keep pushing on.
    """

    generator = torch.Generator().manual_seed(0)
    anchor = torch.randn(8, 768, generator=generator)
    anchor = anchor / anchor.norm(dim=-1, keepdim=True)
    offset = torch.randn(8, 768, generator=generator)
    offset = offset / offset.norm(dim=-1, keepdim=True)
    positive = anchor + 0.30 * offset
    negative = anchor + 0.40 * offset
    return scale * anchor, scale * positive, scale * negative


def test_unnormalised_hinge_goes_vacuous_as_features_grow():
    # Raw squared Euclidean distances scale with the square of the feature
    # norm, but the margin does not. CelebA CLS features and prototypes have
    # norm around 23, so distances land in the hundreds and the paper's margin
    # range of [0.1, 1.0] can no longer keep the hinge open.
    raw = TripletContrastiveLoss(margin=0.5, normalize=False)

    at_unit_scale = raw(*_prototypical_features(1.0)).item()
    at_celeba_scale = raw(*_prototypical_features(23.0)).item()

    assert at_unit_scale > 0.0
    assert at_celeba_scale == 0.0


def test_normalising_keeps_the_hinge_open_at_the_papers_margin():
    normalized = TripletContrastiveLoss(margin=0.5, normalize=True)

    loss = normalized(*_prototypical_features(23.0))

    # Distances now live in [0, 4], so a margin of 0.5 still produces gradients.
    assert loss.item() > 0.0
    assert torch.isfinite(loss)


def test_normalisation_makes_the_loss_scale_invariant():
    normalized = TripletContrastiveLoss(margin=0.5, normalize=True)

    small = normalized(*_prototypical_features(1.0))
    large = normalized(*_prototypical_features(50.0))

    assert small.item() == pytest.approx(large.item(), rel=1e-5)


def test_anchor_weights_reweight_the_hinge():
    anchor = torch.zeros(2, 4)
    positive = torch.tensor([[1.0, 0, 0, 0], [1.0, 0, 0, 0]])
    negative = torch.tensor([[0.0, 1.0, 0, 0], [0.0, 1.0, 0, 0]])
    loss = TripletContrastiveLoss(margin=1.0, normalize=False)

    unweighted = loss(anchor, positive, negative)
    weighted = loss(anchor, positive, negative, weights=torch.tensor([3.0, 1.0]))

    # Both anchors are identical here, so any weighting must agree with the
    # plain mean; this pins the renormalisation rather than the values.
    assert weighted.item() == pytest.approx(unweighted.item())


def test_weights_must_match_the_number_of_anchors():
    anchor, positive, negative = _prototypical_features(1.0)
    loss = TripletContrastiveLoss()

    with pytest.raises(ValueError, match="one entry per anchor"):
        loss(anchor, positive, negative, weights=torch.ones(3))


# --- class balancing of the anchors ----------------------------------------


def _trainer():
    config = FairNetConfig(
        attribute_mode=AttributeMode.FULL,
        hidden_dim=32,
        image_size=32,
        batch_size=8,
        detector_layer=1,
        lora_layers=[2, 3],
        device="cpu",
    )
    backbone = ViTConfig(
        image_size=32,
        patch_size=16,
        num_channels=3,
        hidden_size=32,
        num_hidden_layers=4,
        num_attention_heads=2,
        intermediate_size=32,
    )
    return FairNetTrainer(FairNetViT(backbone, config), config, torch.device("cpu"))


def test_anchor_weights_are_inverse_class_frequencies():
    trainer = _trainer()
    # The shape of CelebA's minority group: many of one class, few of the other.
    labels = torch.tensor([0, 0, 0, 0, 0, 0, 0, 1])

    weights = trainer._anchor_weights(labels)

    assert weights is not None
    assert weights[labels == 0].sum().item() == pytest.approx(weights[labels == 1].sum().item())
    assert weights[-1].item() == pytest.approx(1.0)
    assert weights[0].item() == pytest.approx(1 / 7)


def test_anchor_weights_are_skipped_for_a_single_class_batch():
    trainer = _trainer()

    assert trainer._anchor_weights(torch.zeros(4, dtype=torch.long)) is None


def test_anchor_weighting_can_be_disabled():
    trainer = _trainer()
    trainer.config.class_balanced_contrastive = False

    assert trainer._anchor_weights(torch.tensor([0, 0, 1])) is None


def test_stage4_task_term_is_off_by_default():
    # Stage 4 only sees triggered samples, so an unweighted task loss there
    # reinforces the minority group's own class imbalance.
    assert FairNetConfig().stage4_task_weight == 0.0


def test_class_balanced_stage4_task_loss_equalises_the_classes():
    trainer = _trainer()
    outputs = torch.full((8, 1), 0.5)
    labels = torch.tensor([0, 0, 0, 0, 0, 0, 0, 1])

    balanced = trainer._stage4_task_loss(outputs, labels)
    trainer.config.class_balanced_stage4_task = False
    plain = trainer._task_loss(outputs, labels)

    # At p = 0.5 every sample has the same loss, so balancing must not change
    # the value; this pins the weight normalisation.
    assert balanced.item() == pytest.approx(plain.item(), rel=1e-6)


def test_default_distance_is_plain_euclidean_not_squared():
    # Only plain Euclidean over normalised representations puts the gap
    # d_neg - d_pos near 1.0, which is where the paper's margin range of
    # [0.1, 1.0] actually selects anything.
    anchor = torch.tensor([[1.0, 0.0]])
    negative = torch.tensor([[-1.0, 0.0]])

    default = TripletContrastiveLoss(margin=0.0, normalize=True)
    squared = TripletContrastiveLoss(
        margin=0.0, normalize=True, distance_type="squared_euclidean"
    )

    # d_pos = 0 for both; d_neg is 2 under Euclidean and 4 under its square.
    assert default._compute_distance(anchor, negative).item() == pytest.approx(2.0)
    assert squared._compute_distance(anchor, negative).item() == pytest.approx(4.0)
    assert TripletContrastiveLoss().distance_type == "euclidean"


def test_unknown_distance_type_is_rejected():
    with pytest.raises(ValueError, match="distance_type must be one of"):
        TripletContrastiveLoss(distance_type="manhattan")


def test_euclidean_distance_gradient_is_finite_at_zero_distance():
    anchor = torch.ones(1, 4, requires_grad=True)
    target = torch.ones(1, 4)
    loss = TripletContrastiveLoss(margin=1.0, normalize=False)

    loss(anchor, target, target).backward()

    assert torch.isfinite(anchor.grad).all()
