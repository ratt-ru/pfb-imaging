"""URI helpers that replaced `daskms.fsspec_store.DaskMSStore`.

`DaskMSStore` predates the MSv4 migration. The pipeline only ever used it as a
thin fsspec wrapper -- `.fs.glob`, `.fs.unstrip_protocol`, `.url`, `.exists()`,
`.rm()`, `.makedirs()` -- so `utils/naming.uri_and_fs` and `glob_uris` replace
it and drop a dask-ms (hence python-casacore) dependency from paths that never
needed one.

The equivalence tests below run only where dask-ms is installed: they exist to
pin the claim against the thing they replaced, and the claim is what matters on
an install that cannot have dask-ms at all.
"""

from pathlib import PurePath

import pytest

from pfb_imaging.utils.naming import glob_uris, uri_and_fs

daskms_store = pytest.importorskip("daskms.fsspec_store", reason="dask-ms is behind the [casacore] extra")


@pytest.fixture
def store_tree(tmp_path):
    """Three `.ms` directories and one decoy that must not match `*.ms`."""
    for name in ("a.ms", "b.ms", "c.ms", "notes.txt"):
        target = tmp_path / name
        target.mkdir() if name.endswith(".ms") else target.write_text("x")
    return tmp_path


def test_uri_and_fs_makes_a_relative_path_absolute_and_qualified(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "out.dt").mkdir()

    _, uri = uri_and_fs("out.dt")

    assert uri == f"file://{tmp_path}/out.dt"


@pytest.mark.parametrize(
    "spelling",
    [
        "{p}/out.dt",
        "{p}/out.dt/",
        "file://{p}/out.dt",
    ],
)
def test_uri_and_fs_normalises_equivalent_spellings(tmp_path, spelling):
    """A trailing slash and an explicit protocol both collapse.

    The store URI ends up in log lines and, more importantly, is the string
    handed to zarr -- two spellings of one path must not produce two stores.
    Note `.` segments are NOT collapsed, by fsspec or by DaskMSStore before it.
    """
    (tmp_path / "out.dt").mkdir()

    _, uri = uri_and_fs(spelling.format(p=tmp_path))

    assert uri == f"file://{tmp_path}/out.dt"


def test_uri_and_fs_accepts_a_purepath(tmp_path):
    """The CLI hands paths around as `Path`; DaskMSStore stringified them."""
    (tmp_path / "out.dt").mkdir()

    _, uri = uri_and_fs(PurePath(tmp_path) / "out.dt")

    assert uri == f"file://{tmp_path}/out.dt"


def test_uri_and_fs_keeps_a_remote_protocol():
    """A non-local URL must survive rather than being coerced to a file path."""
    _, uri = uri_and_fs("s3://bucket/some.ms")

    assert uri == "s3://bucket/some.ms"


def test_uri_and_fs_does_not_require_the_path_to_exist(tmp_path):
    """It is called to *create* the .scratch and .dt stores, before they exist."""
    _, uri = uri_and_fs(tmp_path / "not-yet.scratch")

    assert uri == f"file://{tmp_path}/not-yet.scratch"


def test_returned_filesystem_drives_the_store_lifecycle(tmp_path):
    """exists/makedirs/rm on the returned fs, which is what the drivers do."""
    fs, uri = uri_and_fs(tmp_path / "out.scratch")

    assert not fs.exists(uri)
    fs.makedirs(uri, exist_ok=True)
    assert fs.exists(uri)
    (tmp_path / "out.scratch" / "child").mkdir()
    fs.rm(uri, recursive=True)
    assert not fs.exists(uri)


def test_glob_uris_expands_a_pattern_to_qualified_uris(store_tree):
    got = glob_uris(f"{store_tree}/*.ms")

    assert got == [f"file://{store_tree}/{n}" for n in ("a.ms", "b.ms", "c.ms")]


def test_glob_uris_tolerates_a_trailing_slash(store_tree):
    """`--ms foo.ms/` is what tab-completion produces, and used to be rstripped."""
    assert glob_uris(f"{store_tree}/a.ms/") == [f"file://{store_tree}/a.ms"]


def test_glob_uris_returns_empty_rather_than_raising(tmp_path):
    """Callers report "No MS at ..." themselves; they know what they looked for."""
    assert glob_uris(f"{tmp_path}/*.ms") == []


def test_glob_uris_matches_daskmsstore(store_tree):
    """The replacement must expand patterns exactly as DaskMSStore did."""
    pattern = f"{store_tree}/*.ms"
    store = daskms_store.DaskMSStore(pattern)
    legacy = list(map(store.fs.unstrip_protocol, store.fs.glob(pattern)))

    assert glob_uris(pattern) == legacy


@pytest.mark.parametrize(
    "spelling",
    ["{p}", "{p}/", "file://{p}", "{p}/./", "{p}/sub/../sub", "s3://bucket/some.ms"],
)
def test_uri_matches_daskmsstore_url(tmp_path, spelling):
    """`.url` is the string every store call site used; ours must equal it.

    DaskMSStore's `.url` is `fs.unstrip_protocol(get_mapper(url).root)`, and for
    a plain store -- no `path::SUBTABLE` -- that is the same normalisation
    `fsspec.core.url_to_fs` performs. This pins that, including for the s3 case
    the local-path tests cannot reach.
    """
    path = spelling.format(p=tmp_path)

    _, uri = uri_and_fs(path)

    assert uri == daskms_store.DaskMSStore(path).url


def test_uri_and_fs_refuses_a_path_containing_the_chain_separator(tmp_path):
    """`::` must raise, not silently resolve somewhere else.

    fsspec reads `a::b` as a chained URL, so `/data/weird::name.dt` resolves to
    `/data/weird` -- a *different* path that exists and is writable, which is
    how a run would quietly write its .dt to the wrong place. dask-ms read the
    same syntax as CASA subtable selection. Neither reading is ours, so refuse.
    """
    bad = tmp_path / "weird::name.dt"

    with pytest.raises(ValueError, match="chained-URL"):
        uri_and_fs(bad)


def test_fsspec_would_have_silently_truncated_that_path(tmp_path):
    """Pins the hazard the guard above exists for.

    If fsspec ever stops treating `::` as a chain this test fails, which is the
    signal to reconsider the guard rather than leave it as folklore.
    """
    import fsspec

    fs, root = fsspec.core.url_to_fs(str(tmp_path / "weird::name.dt"))

    assert fs.unstrip_protocol(root) == f"file://{tmp_path}/weird"
