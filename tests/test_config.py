import pytest

from fairnet.config import AttributeMode, FairNetConfig


def test_string_attribute_mode_is_normalized():
    config = FairNetConfig(attribute_mode="unlabeled")

    assert config.attribute_mode is AttributeMode.UNLABELED


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"lora_rank": 0}, "positive"),
        ({"lora_dropout": 1.0}, "lora_dropout"),
        ({"sensitive_attributes": [9, 9]}, "duplicates"),
        ({"sensitive_attributes": [20]}, "cannot also"),
        ({"lora_layers": [1, 1]}, "duplicates"),
    ],
)
def test_invalid_configuration_has_actionable_error(kwargs, message):
    with pytest.raises(ValueError, match=message):
        FairNetConfig(**kwargs)
