from firecast.observability.experiment import sha256_file


def test_sha256_file_is_stable(tmp_path):
    source = tmp_path / "dataset.csv"
    source.write_text("x,y\n1,0\n", encoding="utf-8")
    assert sha256_file(source) == sha256_file(source)
