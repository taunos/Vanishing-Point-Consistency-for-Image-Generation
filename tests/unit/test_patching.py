from pcs.regional.patching import generate_overlapping_grid_patches


def test_generate_overlapping_grid_patches_count_and_bounds() -> None:
    patches = generate_overlapping_grid_patches(
        image_width=120,
        image_height=80,
        scales=[2, 3, 4],
        overlap_ratio=0.2,
        min_patch_size=1,
    )

    assert len(patches) == (2 * 2) + (3 * 3) + (4 * 4)
    assert len({patch.patch_id for patch in patches}) == len(patches)
    assert patches[0].patch_id == "s2x2_r00_c00_o20"

    for patch in patches:
        assert 0 <= patch.x0 < patch.x1 <= 120
        assert 0 <= patch.y0 < patch.y1 <= 80


def test_generate_overlapping_grid_patches_filters_tiny_patches() -> None:
    patches = generate_overlapping_grid_patches(
        image_width=20,
        image_height=20,
        scales=[4],
        overlap_ratio=0.0,
        min_patch_size=8,
    )

    assert patches == []

