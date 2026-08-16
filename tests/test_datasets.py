import torch

from fairnet.datasets import SyntheticBiasedDataset, create_synthetic_loaders


def test_synthetic_dataset_is_lazy_deterministic_and_contains_both_groups():
    dataset = SyntheticBiasedDataset(
        num_samples=10,
        minority_ratio=0.2,
        image_size=8,
        seed=7,
    )

    first_image, first_attributes = dataset[3]
    second_image, second_attributes = dataset[3]

    assert torch.equal(first_image, second_image)
    assert torch.equal(first_attributes, second_attributes)
    assert first_image.shape == (3, 8, 8)
    assert set(dataset.sensitive.tolist()) == {0, 1}
    assert not hasattr(dataset, "images")


def test_small_synthetic_loader_does_not_drop_every_sample():
    train, validation, test = create_synthetic_loaders(
        batch_size=16,
        num_train=4,
        num_val=4,
        num_test=4,
        image_size=8,
    )

    assert len(train) == len(validation) == len(test) == 1
