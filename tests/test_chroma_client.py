def test_project_hash_is_stable():
    from chroma_client import project_hash
    h1 = project_hash("/my/project")
    h2 = project_hash("/my/project")
    assert h1 == h2
    assert len(h1) == 8


def test_project_hash_differs_for_different_paths():
    from chroma_client import project_hash
    assert project_hash("/project/a") != project_hash("/project/b")


def test_get_collection_creates_and_reuses(tmp_data_dir):
    import importlib
    import chroma_client
    importlib.reload(chroma_client)
    col1 = chroma_client.get_collection("code", "/my/project")
    col2 = chroma_client.get_collection("code", "/my/project")
    assert col1.name == col2.name


def test_different_prefixes_give_different_collections(tmp_data_dir):
    import importlib
    import chroma_client
    importlib.reload(chroma_client)
    c1 = chroma_client.get_collection("code", "/my/project")
    c2 = chroma_client.get_collection("notes", "/my/project")
    assert c1.name != c2.name
