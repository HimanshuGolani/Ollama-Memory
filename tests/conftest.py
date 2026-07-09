import pytest

@pytest.fixture
def tmp_data_dir(tmp_path, monkeypatch):
    import config
    monkeypatch.setattr(config.settings, "data_dir", tmp_path)
    return tmp_path
