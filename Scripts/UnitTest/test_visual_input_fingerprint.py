import torch

from Utility.VisualInputFingerprint import visual_input_sha256


def test_visual_input_fingerprint_is_stable_and_value_sensitive():
    fields = {
        "pixel2_uv": torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32),
        "pixel2_disp": torch.tensor([[5.0], [6.0]], dtype=torch.float32),
        "pixel2_uv_cov": torch.tensor([[0.1, 0.2, 0.0], [0.3, 0.4, 0.0]], dtype=torch.float32),
    }
    same_values = {name: value.clone() for name, value in reversed(list(fields.items()))}
    changed = {name: value.clone() for name, value in fields.items()}
    changed["pixel2_uv"][0, 0] += 0.01

    assert visual_input_sha256(fields) == visual_input_sha256(same_values)
    assert visual_input_sha256(fields) != visual_input_sha256(changed)


def test_visual_input_fingerprint_distinguishes_shape_and_dtype():
    values = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32)

    flat = visual_input_sha256({"pixel2_uv": values})
    reshaped = visual_input_sha256({"pixel2_uv": values.reshape(2, 2)})
    double = visual_input_sha256({"pixel2_uv": values.double()})

    assert flat != reshaped
    assert flat != double
