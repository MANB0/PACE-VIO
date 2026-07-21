import csv
from pathlib import Path

from Scripts.analyse_clear_circle_pair_vio import make_trace_label, read_manifests


def _write_manifest(root: Path, scene: str, variant: str) -> None:
    root.mkdir(parents=True)
    with (root / "run_manifest.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["scene", "variant", "result_dir", "scene_root"])
        writer.writeheader()
        writer.writerow(
            {
                "scene": scene,
                "variant": variant,
                "result_dir": str(root / "result"),
                "scene_root": str(root / "scene"),
            }
        )


def test_read_manifests_combines_multiple_result_roots(tmp_path: Path) -> None:
    vio_root = tmp_path / "vio"
    pure_root = tmp_path / "pure"
    _write_manifest(vio_root, "clear_circle_zero_noise", "vio_preintegrated_full")
    _write_manifest(pure_root, "clear_circle_zero_noise", "pure_macvo")

    rows = read_manifests([vio_root, pure_root])

    assert [(row["scene"], row["variant"]) for row in rows] == [
        ("clear_circle_zero_noise", "vio_preintegrated_full"),
        ("clear_circle_zero_noise", "pure_macvo"),
    ]


def test_make_trace_label_separates_variants_for_same_scene() -> None:
    assert (
        make_trace_label({"scene": "clear_circle_zero_noise", "variant": "pure_macvo"})
        == "clear_circle_zero_noise / pure_macvo"
    )
