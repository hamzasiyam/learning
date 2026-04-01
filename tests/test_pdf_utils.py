from modules.core.pdf_utils import meta_to_pdf_metadata, split_images_by_size


def test_meta_to_pdf_metadata_maps_fields():
    meta = {
        "title": "My Title",
        "author": "Me",
        "tags": "a,b",
        "type": "notes",
        "published": "2026-03-19",
        "language": "en",
    }
    out = meta_to_pdf_metadata(meta)
    assert out["/Title"] == "My Title"
    assert out["/Author"] == "Me"
    assert out["/Keywords"] == "a,b"
    assert out["/Type"] == "notes"
    assert out["/CreationDate"] == "D:20260319000000"
    assert out["/Language"] == "en"


def test_split_images_by_size_creates_at_least_one_batch(tmp_path):
    img1 = tmp_path / "a.jpg"
    img2 = tmp_path / "b.jpg"
    img1.write_bytes(b"x" * 200_000)
    img2.write_bytes(b"y" * 200_000)

    batches = split_images_by_size([str(img1), str(img2)], target_size_mb=0.1, optimize=False)
    assert len(batches) >= 1
    assert sum(len(b) for b in batches) == 2

