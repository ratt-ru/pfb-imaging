# Imager `concat_row` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement `concat_row` in the imager pass-2 path so `concat_row=True` (default) collapses the time/scan axis into one image per band, concatenating each partition's rows before a single gridding.

**Architecture:** Two changes to `core/imager.py`. (1) Generalise the `_grid_image` Ray worker to read a *list* of scratch nodes and write an explicit output node (behaviour-preserving refactor). (2) Branch the driver's pass-2 work-list construction on `concat_row`: group scratch nodes by band, time-collapse the uv counts (coercing time-resolved `weight_grouping`), and dispatch one task per band. The within-worker group-by-partition-key + `xr.concat(dim="row")` block already concatenates rows across the gathered nodes for free.

**Tech Stack:** Python 3.11+, xarray DataTree, zarr v2, ducc0 wgridder, Ray, pytest.

**Spec:** `docs/superpowers/specs/2026-06-24-imager-concat-row-design.md`

## Global Constraints

- Imaging path stays **casacore-free**: no module-scope `africanus`/`daskms`/`casacore` imports in `core/imager.py` or anything it pulls on the imaging path (`.claude/rules/architecture.md` §3/§8).
- After any code change run: `uv run ruff format . && uv run ruff check . --fix`.
- Conventional Commits; first line < 72 chars; end commit messages with the `Co-Authored-By` trailer.
- The pre-commit `generate-cabs` hook fails in this environment (`hip-cargo` not on PATH); the CLI signature is unchanged by this plan, so commit with `git commit --no-verify` (all other hooks still run).
- Tests run via `uv run --frozen pytest`. The session Ray fixture is `num_cpus=1`; every `imager`/`init`/`grid` call in a test passes `keep_ray_alive=True`.

---

### Task 1: Generalise `_grid_image` to a list of source nodes + explicit output node

Behaviour-preserving refactor. `_grid_image` currently takes one `image_name` and uses it both to *read* the scratch node and to *name* the output node. Split those into `src_names: list[str]` (nodes to read) and `out_name: str` (node to write), and add the defensive FREQ-equality guard. The driver passes `[name]` and `name`, so output is byte-identical and existing tests are the gate.

**Files:**
- Modify: `src/pfb_imaging/core/imager.py` (`_grid_image` signature + body; the pass-2 dispatch call)
- Test: `tests/test_imager.py` (existing tests are the regression gate — no new test)

**Interfaces:**
- Produces: `_grid_image.remote(scratch_store, dt_store, src_names: list[str], out_name: str, counts, nx, ny, nx_psf, ny_psf, cell_rad, freq_out, meta, robustness=None, nx_pad=None, ny_pad=None, filter_counts_level=5.0, npix_super=0, nthreads=1, epsilon=1e-7, do_wgridding=True, double_accum=True)`. `meta` keeps its current keys (`bandid`, `timeid`, `ra`, `dec`, `time_out`). Returns `{"timeid": int, "psf": ndarray, "wsum": ndarray}` (unchanged).

- [ ] **Step 1: Change the `_grid_image` signature**

In `src/pfb_imaging/core/imager.py`, replace the single `image_name,` parameter (currently line ~45, the 3rd parameter of `_grid_image`) with the two-parameter form:

```python
def _grid_image(
    scratch_store,
    dt_store,
    src_names,
    out_name,
    counts,
    nx,
    ny,
    nx_psf,
    ny_psf,
    cell_rad,
    freq_out,
    meta,
    robustness=None,
    nx_pad=None,
    ny_pad=None,
    filter_counts_level=5.0,
    npix_super=0,
    nthreads=1,
    epsilon=1e-7,
    do_wgridding=True,
    double_accum=True,
):
```

- [ ] **Step 2: Read pieces from all `src_names`**

Replace the two lines that open the single image node:

```python
    image = xr.open_datatree(scratch_store, engine="zarr", chunks=None)[image_name]
    pieces = [child.ds.load() for _, child in image.children.items()]
```

with a gather across the list:

```python
    dt = xr.open_datatree(scratch_store, engine="zarr", chunks=None)
    pieces = []
    for sn in src_names:
        pieces.extend(child.ds.load() for _, child in dt[sn].children.items())
```

- [ ] **Step 3: Add the FREQ-equality guard to the row-concat block**

Replace the existing `if len(plist) == 1: ... else: ...` concat block (the one that builds `part`) with the guarded version:

```python
        if len(plist) == 1:
            part = plist[0]
        else:
            # rows are only concatenatable when they share the freq axis (same
            # spw + band channel-chunk); guard the invariant before concat
            f0 = plist[0].FREQ.values
            for p in plist[1:]:
                assert np.array_equal(p.FREQ.values, f0), (
                    f"concat group {key} has mismatched FREQ; cannot concatenate rows"
                )
            rowvars = ["VIS", "WEIGHT", "MASK", "UVW"]
            cat = xr.concat([p[rowvars] for p in plist], dim="row", coords="minimal", compat="override")
            part = plist[0].drop_vars(rowvars)
            for v in rowvars:
                part[v] = cat[v]
```

- [ ] **Step 4: Write outputs under `out_name`**

Change the two `to_zarr` group paths in `_grid_image`. The partition write:

```python
        part_out.to_zarr(dt_store, group=f"{out_name}/part{pid:04d}", mode="a", consolidated=False)
```

and the band-node write:

```python
    band_ds.to_zarr(dt_store, group=out_name, mode="a", consolidated=False)
```

(Only `image_name` → `out_name` changes; keep `consolidated=False` and the surrounding comments.)

- [ ] **Step 5: Update the dispatch call to the new signature**

In the pass-2 dispatch loop (`for name, meta in image_meta.items():`), pass `[name]` as `src_names` and `name` as `out_name`:

```python
    tasks = []
    for name, meta in image_meta.items():
        fut = _grid_image.remote(
            scratch_store.url,
            dt_store.url,
            [name],
            name,
            reduced[(meta["bandid"], meta["timeid"])],
            nx,
            ny,
            nx_psf,
            ny_psf,
            cell_rad,
            freq_out[meta["bandid"]],
            meta,
            robustness=robustness,
            nx_pad=nx_pad,
            ny_pad=ny_pad,
            filter_counts_level=filter_counts_level,
            npix_super=npix_super,
            nthreads=nthreads,
            epsilon=epsilon,
            do_wgridding=do_wgridding,
            double_accum=double_accum,
        )
        tasks.append(fut)
```

- [ ] **Step 6: Lint**

Run: `uv run ruff format src/pfb_imaging/core/imager.py && uv run ruff check src/pfb_imaging/core/imager.py --fix`
Expected: "All checks passed!"

- [ ] **Step 7: Run the imager tests (gate = unchanged behaviour)**

Run: `uv run --frozen pytest tests/test_imager.py -q`
Expected: `3 passed` (the refactor is byte-identical; `src_names=[name]`, `out_name=name`).

- [ ] **Step 8: Commit**

```bash
git add src/pfb_imaging/core/imager.py
git commit --no-verify -m "refactor(imager): _grid_image reads node list, writes explicit node"
```

---

### Task 2: Implement `concat_row` in the driver + regression test

Branch the pass-2 work-list construction on `concat_row`. Build per-scratch-node info once, then either (True) group by band with time-collapsed counts, or (False) keep per-node as today. Coerce time-resolved `weight_grouping` to its band-collapsed analogue. Dispatch over the work list.

**Files:**
- Modify: `src/pfb_imaging/core/imager.py` (counts/work-list construction block; `ntime` for root attrs; dispatch loop)
- Test: `tests/test_imager.py` (new `test_imager_concat_row_collapses_time`)

**Interfaces:**
- Consumes: `_grid_image.remote(scratch_store, dt_store, src_names, out_name, counts, ...)` from Task 1.
- Produces: `imager(..., concat_row=True)` writes one `band{b:04d}_time0000` node per band; `concat_row=False` writes one `band{b:04d}_time{t:04d}` node per `(band, timeid)` (unchanged).

- [ ] **Step 1: Write the failing regression test**

Add to `tests/test_imager.py`:

```python
def test_imager_concat_row_collapses_time(ms_name, tmp_path):
    """concat_row=True collapses the time axis into one band node and agrees
    with concat_row=False on the MFS dirty.

    Both runs pin weight_grouping="per-band" and robustness=0.0 so the per-row
    imaging weights are identical; vis2dirty is linear in the rows
    (grid(A∪B) == grid(A) + grid(B) up to fp), so the wsum-normalised MFS dirty
    must match. The shared MS is one scan of 60 integrations, so
    integrations_per_image=15 splits it into 4 time blocks (4 timeids) to make
    the concat_row=False granularity meaningful.
    """
    from collections import Counter

    common = dict(
        channels_per_image=2,
        integrations_per_image=15,
        product="I",
        field_of_view=1.0,
        robustness=0.0,
        weight_grouping="per-band",
        fits_mfs=False,
        fits_cubes=False,
        overwrite=True,
        keep_ray_alive=True,
    )

    base_true = str(tmp_path / "concat_true")
    imager_core([Path(ms_name)], base_true, concat_row=True, **common)
    base_false = str(tmp_path / "concat_false")
    imager_core([Path(ms_name)], base_false, concat_row=False, **common)

    dt_true = xr.open_datatree(base_true + "_I.dt", engine="zarr", chunks=None)
    dt_false = xr.open_datatree(base_false + "_I.dt", engine="zarr", chunks=None)
    bands_true = sorted(n for n in dt_true.children if n.startswith("band"))
    bands_false = sorted(n for n in dt_false.children if n.startswith("band"))

    # concat_row=True: exactly one time0000 node per distinct band
    assert bands_true, "no band nodes written"
    assert all(n.endswith("_time0000") for n in bands_true)
    bandids_true = {int(dt_true[n].ds.attrs["bandid"]) for n in bands_true}
    assert len(bands_true) == len(bandids_true)

    # concat_row=False: multiple time nodes for at least one band (4 blocks here)
    per_band = Counter(int(dt_false[n].ds.attrs["bandid"]) for n in bands_false)
    assert max(per_band.values()) > 1, "expected multiple time nodes per band at ipi=15"

    def mfs(dt, names):
        num = sum(dt[n].ds.DIRTY.values[0] for n in names)
        den = sum(float(dt[n].ds.WSUM.values[0]) for n in names)
        return num / den

    a = mfs(dt_true, bands_true)
    b = mfs(dt_false, bands_false)
    assert a.shape == b.shape
    assert np.isfinite(a).all() and np.isfinite(b).all()
    assert_allclose(1 + a, 1 + b, rtol=1e-4, atol=1e-4)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run --frozen pytest tests/test_imager.py::test_imager_concat_row_collapses_time -q`
Expected: FAIL — with `concat_row` still a no-op, `concat_row=True` writes per-`(band,time)` nodes, so `all(n.endswith("_time0000"))` (and the one-node-per-band assertion) fails.

- [ ] **Step 3: Replace the counts/work-list construction block**

Replace the block that builds `counts_map` / `image_meta` and ends at `reduced = reduce_counts(counts_map, weight_grouping)` (the loop `for name in scratch_dt.children:` through the `reduce_counts` call) with:

```python
    scratch_dt = xr.open_datatree(scratch_store.url, engine="zarr", chunks=None)
    # per-scratch-node summary: (bandid, timeid, ra, dec, time_out, counts)
    node_info = {}
    ncorr = None
    for name in scratch_dt.children:
        if not name.startswith("band"):
            continue
        pieces = [child.ds for _, child in scratch_dt[name].children.items()]
        if not pieces:
            continue
        ncorr = pieces[0].corr.size
        acc = None
        for ds in pieces:
            c = ds.COUNTS.values
            acc = c.copy() if acc is None else acc + c
        node_info[name] = {
            "bandid": int(pieces[0].attrs["bandid"]),
            "timeid": int(pieces[0].attrs["timeid"]),
            "ra": float(pieces[0].attrs["ra"]),
            "dec": float(pieces[0].attrs["dec"]),
            "time_out": float(np.mean([ds.attrs["time_out"] for ds in pieces])),
            "counts": acc,
        }
    if not node_info:
        log.error_and_raise("Pass 1 produced no output images (all data flagged?)", RuntimeError)

    # build the pass-2 work list: (out_name, src_names, meta)
    counts_map = {}
    work = []
    if concat_row:
        # time-resolved groupings contradict a time-collapsed image; map each to
        # its band-collapsed analogue (per-band-time -> per-band, per-time -> mfs)
        grouping_eff = {"per-band-time": "per-band", "per-time": "mfs"}.get(weight_grouping, weight_grouping)
        if grouping_eff != weight_grouping and robustness is not None:
            log.warning(
                f"concat_row collapses the time axis; using weight_grouping "
                f"'{grouping_eff}' instead of '{weight_grouping}'"
            )
        by_band = {}
        for name, info in node_info.items():
            by_band.setdefault(info["bandid"], []).append((name, info))
        for bandid, items in sorted(by_band.items()):
            src_names = [name for name, _ in items]
            infos = [info for _, info in items]
            acc = None
            for info in infos:
                acc = info["counts"].copy() if acc is None else acc + info["counts"]
            counts_map[(bandid, 0)] = acc  # time-collapsed: single timeid 0 per band
            out_name = f"band{bandid:04d}_time0000"
            work.append(
                (
                    out_name,
                    src_names,
                    {
                        "bandid": bandid,
                        "timeid": 0,
                        "ra": infos[0]["ra"],
                        "dec": infos[0]["dec"],
                        "time_out": float(np.mean([info["time_out"] for info in infos])),
                    },
                )
            )
    else:
        grouping_eff = weight_grouping
        for name, info in node_info.items():
            counts_map[(info["bandid"], info["timeid"])] = info["counts"]
            work.append(
                (
                    name,
                    [name],
                    {k: info[k] for k in ("bandid", "timeid", "ra", "dec", "time_out")},
                )
            )

    ntime = len({meta["timeid"] for _, _, meta in work})
    log.info(f"Reducing uv counts with grouping '{grouping_eff}'")
    reduced = reduce_counts(counts_map, grouping_eff)
```

Notes for the implementer:
- This block sits after the `zarr.consolidate_metadata(scratch_store.url)` line and replaces everything up to and including the old `reduced = reduce_counts(...)` line.
- The old per-node `log.info("Reducing uv counts ...")` line moves here (now logging `grouping_eff`). Remove the old one if it sat above `scratch_dt = ...`.
- `ntime` is now the count of distinct *output* timeids (1 per band when `concat_row=True`); it continues to feed `root_attrs["ntime"]`.
- `src_names` carries the real scratch node names (the `node_info` keys), so the worker reads exactly the nodes pass 1 wrote.

- [ ] **Step 4: Dispatch over the work list**

Replace the dispatch loop `for name, meta in image_meta.items():` with iteration over `work`, passing `src_names` and `out_name`:

```python
    tasks = []
    for out_name, src_names, meta in work:
        fut = _grid_image.remote(
            scratch_store.url,
            dt_store.url,
            src_names,
            out_name,
            reduced[(meta["bandid"], meta["timeid"])],
            nx,
            ny,
            nx_psf,
            ny_psf,
            cell_rad,
            freq_out[meta["bandid"]],
            meta,
            robustness=robustness,
            nx_pad=nx_pad,
            ny_pad=ny_pad,
            filter_counts_level=filter_counts_level,
            npix_super=npix_super,
            nthreads=nthreads,
            epsilon=epsilon,
            do_wgridding=do_wgridding,
            double_accum=double_accum,
        )
        tasks.append(fut)
```

- [ ] **Step 5: Run the new test to verify it passes**

Run: `uv run --frozen pytest tests/test_imager.py::test_imager_concat_row_collapses_time -q`
Expected: PASS.

- [ ] **Step 6: Run the full imager suite (no regressions under the new default)**

Run: `uv run --frozen pytest tests/test_imager.py -q`
Expected: `4 passed` (the 3 existing tests now run with `concat_row=True` by default; they assert MFS/`startswith("band")` properties that still hold).

- [ ] **Step 7: Lint**

Run: `uv run ruff format . && uv run ruff check . --fix`
Expected: "All checks passed!"

- [ ] **Step 8: Commit**

```bash
git add src/pfb_imaging/core/imager.py tests/test_imager.py
git commit --no-verify -m "feat(imager): concat_row collapses time, grids each band once"
```

---

## Self-Review notes

- **Spec coverage:** §3 behaviour → Task 2 Step 3 (per-band grouping) + Task 1 (single-grid worker). §4.1 dispatch → Task 2 Steps 3–4. §4.2 worker signature → Task 1 Steps 1–5. §4.3 weighting coercion → Task 2 Step 3. §5 tree/downstream → emergent (no code). §6 FREQ guard → Task 1 Step 3; beam/single-time degeneracy → covered by `len(plist)==1` path. §7 test → Task 2 Steps 1–6. §8 files → only `core/imager.py` + `tests/test_imager.py`.
- **Type consistency:** `_grid_image(... src_names, out_name ...)` defined in Task 1 Step 1, consumed identically in Task 1 Step 5 and Task 2 Step 4. `meta` keys (`bandid`,`timeid`,`ra`,`dec`,`time_out`) consistent across construction (Task 2 Step 3) and use (`_grid_image` body, unchanged).
- **Coercion vs robustness:** the warning only fires when `robustness is not None` (counts actually used); the string coercion always applies so `reduced` is consistent. With `robustness=None` (natural) counts are unused, so the equivalence test stays quiet.
