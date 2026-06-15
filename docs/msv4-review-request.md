# Review request — `pfb imager` MSv4 DataTree pipeline (PR #252)

> **For the reviewer (arcae / xarray-ms / xarray-kat author):** this PR adds an
> MSv4 front-end (`pfb imager`) that reads visibilities via the `arcae`
> `xarray-ms` engine and writes a unified `xarray.DataTree`. I'd most value your
> take on **(1) the arcae ⊥ python-casacore process separation** and **(2)
> whether our MSv4 / xarray-ms access patterns are idiomatic and robust**. The
> gridding/deconv maths is already validated against the legacy `init`+`grid`
> path (matches to ~1e-14); the parts where I'm least sure are the ones that
> touch your libraries. Priority reading order and specific questions below.

> **Status (2026-06-15): review complete — sjperkins' responses are inlined below**
> as **A (sjperkins):** blocks. Headline outcome: **Focus 1 is resolved** — as of
> **arcae 0.5.2** (ratt-ru/arcae#211, ratt-ru/arcae#212) arcae and python-casacore
> coexist in a single process, so the process-separation guardrails (two-invocation
> test split, subprocess equivalence test, CI/doc notes) are being removed. The
> imaging path stays casacore-free *by choice* (lightweight install / fast startup),
> not out of necessity.

## What the PR does (1 minute)

`pfb imager` is a two-pass, Ray-distributed pipeline:
1. **Pass 1** reads raw MSv4 finely (per scan), converts to Stokes, averages,
   and writes per-piece vis + a uv `COUNTS` grid into a `.scratch` DataTree.
2. **Pass 2** groups pieces into partitions, grids each with ducc0, and writes a
   single `<out>_<P>.dt` DataTree (one node per `(band,time)` output image, one
   `part####` child per data partition).

It replaces the legacy `.xds`+`.dds` split *for this path only*; `init`/`grid`
(python-casacore) remain live and are the equivalence reference.

---

## Priority reading order

### P0 — the two areas I want your eyes on

| File | What to look at |
|---|---|
| `src/pfb_imaging/core/imager.py` | `get_engine()` (engine/backend dispatch + `partition_schema`); the two `xr.open_datatree` node-iteration loops; the missing-row / NaN handling (lines ~381–390). |
| `src/pfb_imaging/utils/stokes2vis_msv4.py` | The bulk of the MSv4 schema assumptions — per-node access, antenna reconstruction, regular-vs-irregular handling, phase centre, polarization. |
| `tests/conftest.py` | The arcae⊥casacore **and** uv⊥Ray isolation (`RAY_ENABLE_UV_RUN_RUNTIME_ENV=0`, the session Ray fixture). |
| `.claude/rules/architecture.md` §3, §8 · `.claude/rules/testing-and-ci.md` §1 | The documented import discipline and the two-invocation test split. |

### P1 — consumes the products (less arcae-specific)
- `src/pfb_imaging/utils/fits.py` — `dt2fits`/`rdt2fits` (casacore-free FITS; time via `utils/misc.to_unix_time` instead of `casacore.quanta`).
- `src/pfb_imaging/operators/gridder.py` — `grid_partition`, `residual_from_partitions`.
- `src/pfb_imaging/operators/hessian.py` — `HessianTree`.
- `src/pfb_imaging/utils/{misc,beam}.py` — the *deferred* `daskms`/`africanus` imports that keep the imaging path casacore-free.

### P2 — skim
- `src/pfb_imaging/cli/imager.py`, `src/pfb_imaging/utils/weighting.py` (`reduce_counts`).
- Tests: `tests/test_imager.py` (arcae-only invocation + subprocess equivalence), `test_imager_pass2.py`, `test_hessian_tree.py`, `test_fits_tree.py` (all casacore-free).

---

## Focus 1 — arcae ⊥ python-casacore separation

We treat arcae and python-casacore as **process-exclusive** (arcae#72): the
entire imaging path is kept casacore-free so arcae and ducc0 coexist, and
casacore-pulling imports (`africanus`, `daskms`, `casacore`) are deferred into
the functions that need them rather than imported at module scope. Consequences:
- `from scipy.constants import c as lightspeed` (not `africanus.constants`); a
  local `to_unix_time` (not `casacore.quanta.quantity`) for MJD→unix in FITS.
- Tests split into two pytest invocations — `tests/test_imager.py` (arcae) alone,
  everything else with `--ignore=tests/test_imager.py` (casacore). Running them
  together segfaults.
- The imager↔`init`+`grid` equivalence test runs the casacore reference in a
  **subprocess** so the arcae process never imports casacore.
- `conftest.py` sets `RAY_ENABLE_UV_RUN_RUNTIME_ENV=0` before importing Ray:
  under `uv run`, Ray's uv hook otherwise rebuilds a per-worker venv (missing the
  `[full]` extra) so workers crash on `import ray`.

**Questions:**
1. Is full process separation still the right model, or is there a supported path
   to arcae + python-casacore coexistence we should track instead?

   > **A (sjperkins):** Initially — "Yes, process separation is the correct model."
   >
   > **Update (superseding):** "The following PRs included in **arcae 0.5.2** should
   > now make this process separation redundant — ratt-ru/arcae#211, ratt-ru/arcae#212.
   > arcae and python-casacore can now co-exist in the same process." → We are
   > therefore removing the test/CI/docs guardrails; the imaging path stays
   > casacore-free only as a lightweight-install / startup choice.

2. Could anything in the arcae/`xarray-ms` import chain pull python-casacore
   transitively (so our "casacore-free" guarantee is weaker than we think)?

   > **A (sjperkins):** "No — xarray-ms/arcae never import python-casacore."

3. Is disabling `RAY_ENABLE_UV_RUN_RUNTIME_ENV` the sanctioned way to run arcae
   inside Ray workers under `uv`, or is there a cleaner pattern?

   > **A (sjperkins):** This flag is **orthogonal** to the arcae ⊥ casacore question.
   > It controls whether Ray propagates the `uv run` environment to workers (when set,
   > Ray deactivates the regular `RAY_RUNTIME_ENV_HOOK`); it matters mainly for local
   > clusters that may share an environment. Both libraries can be installed in one
   > venv — only *importing both in one process* caused the (now-fixed) symbol clash.
   > Suggests a Ray mini-cluster (`ray.cluster_utils.Cluster`) for tests. → This
   > setting therefore **stays** in `conftest.py` (it is not an arcae⊥casacore guardrail).

## Focus 2 — MSv4 / xarray-ms access patterns

Each pattern below is something we'd like sanity-checked for idiom + robustness.
File references are `core/imager.py` (CI = `imager.py`) and
`utils/stokes2vis_msv4.py` (= `s2v`).

1. **Engine dispatch** — `get_engine()` (`imager.py`): `CASA_TABLE → "xarray-ms:msv2"` with
   `partition_schema=["FIELD_ID","DATA_DESC_ID","SCAN_NUMBER"]`; `ZARR → "zarr"`;
   `MEERKAT → "xarray-kat"` (`applycal="all"`, `uvw_sign_convention="casa"`).
   *Is this partition schema the right granularity? Are we handling DDID vs SPW correctly?*

   > **A (sjperkins):** These partition values are probably suitable for MSs produced by
   > `msv4toms.py`. MSs from other instruments may need to be partitioned by other columns
   > (`SOURCE_ID` is the obvious one). See the
   > [xarray-ms partitioning docs](https://xarray-ms.readthedocs.io/en/latest/partitioning.html).
   > *(On the `MEERKAT`/`applycal`/`uvw_sign_convention` setup — "fine for the time being,
   > but a bit ugly; cf. @bennahugo's similar setup in tricolour — worth a nicer approach
   > later. We can always refactor.")*

2. **Visibility-node filter** — `node.attrs.get("type") not in VISIBILITY_XDS_TYPES`
   (from `msv4_utils`). *Is this the canonical way to enumerate visibility nodes in a DataTree?*

   > **A (sjperkins):** "Briefly, yes. A more nuanced answer: no canonical MSv4 traversal
   > methods have been defined yet."

3. **DATA → VISIBILITY rename** (`imager.py` ~343–347): we map the `DATA` column name to
   `VISIBILITY`. *Always correct? Should we be going through `data_groups` / the
   `correlated_data` field instead of assuming the name?*

   > **A (sjperkins):** "To be fully compliant, one should go through the `data_groups`
   > mechanism. Probably the user/developer would have to supply the `data_group` name to
   > perform the lookup."

4. **Per-node identity** (`s2v:77–79`): `field_name` / `scan_name` via
   `np.unique(...).item()`, SPW via `ds.frequency.attrs["spectral_window_name"]`.
   *Robust given the partition schema, or fragile?*

   > **A (sjperkins):** "`field_names` and `scan_names` can technically vary per partition.
   > Multiple values in `field_names` imply multiple entries for that field in
   > `field_and_source_{data_group_name}_xds`. The complexity of supporting this at the
   > partition level may not be worth it, especially if the xarray-ms `PARTITION_SCHEMA`
   > includes `FIELD_ID` and `SCAN_NUMBER` by default. xarray-kat scans only point at a
   > single source, so this holds for MeerKAT archive data. The clarifying question is
   > whether pfb should attempt to image MSs that have been substantially modified by CASA
   > tasks."

5. **ANTENNA1/2 reconstruction** (`s2v:119–136`): we rebuild MSv2-style integer
   `ant1`/`ant2` from `baseline_antenna1_name`/`baseline_antenna2_name` via
   `np.unique(..., return_inverse=True)`. There's a commented-out alternative using
   `antenna_xds.antenna_name` + `searchsorted`. *Which is correct? Does index
   ordering (name-sorted vs antenna_xds order) matter downstream (it feeds averaging
   + gains)?*

   > **A (sjperkins):** "The `np.unique` approach might not map the resultant antenna indices
   > to the entries in the `antenna_xds` subtable in all cases. It's probably sufficient when
   > calling MSv2 code that only needs to establish baseline relations with integer indices
   > inside high-performance kernels (numba / C/C++). If `antenna_xds` variables (e.g.
   > `ANTENNA_POSITION`) are passed into the kernels, then the `np.searchsorted` approach is
   > required."

6. **Regular vs irregular discriminator** (`s2v:114`, `:231`): we treat
   `INTEGRATION_TIME` / `CHANNEL_WIDTH` as present in `ds.data_vars` when irregular,
   else read the value from the coord attrs (`ds.time.attrs["integration_time"]`,
   `ds.frequency.attrs["channel_width"]`). *Is `"X" in ds.data_vars` a reliable
   regular/irregular signal?*

   > **A (sjperkins):** "xarray-ms signals this case with `nan` in the `integration_time` and
   > `channel_width` attrs. xarray-ms may be too pedantic here — I'm currently discussing this
   > with the MSv4 working group. It may be that we should reason about time using
   > `TIME_CENTROID` and `EFFECTIVE_INTEGRATION_TIME` (and the related variables for
   > frequency)."

7. **Missing-row / NaN handling** (`imager.py:381–390`, `s2v` flag path): we note that
   `xarray-ms` fills a regular grid over irregular/missing data with **NaNs** (vs
   `xarray-kat`, which zeroes vis+weights), and we detect missing rows with
   `np.isnan(UVW).all(axis=-1)`. *Is NaN-in-UVW the canonical "missing row" marker?
   Can `VISIBILITY` be NaN while `UVW` is finite (or vice versa)? Is there a
   first-class row-presence / flag indicator we should use instead of sniffing NaNs?*
   This one bit us indirectly (a zero-DATA test MS), so we care about getting the
   "absent data" contract right.

   > **A (sjperkins):** "NaN is the canonical missing value for floating-point data
   > ([xarray docs](https://docs.xarray.dev/en/v2025.03.0/user-guide/computation.html#missing-values)),
   > and xarray-ms uses it when establishing a regular grid over an irregular one. By contrast
   > the xarray-kat grid is always regular, but (1) some chunks may not have been written
   > (especially at the start of an observation) and (2) if network-access max-retries fail,
   > xarray-kat returns missing values for the chunk. xarray-kat currently *zeroes* missing
   > visibilities and weights — I'm very open to changing these to NaN to synchronise with
   > xarray's missing-value mechanisms."

8. **Phase centre** (`s2v:172–173`): `data_groups["base"]["field_and_source"]` →
   `FIELD_PHASE_CENTER_DIRECTION.values[0]`. *Is assuming the `"base"` data group
   safe? Multiple data groups?*

   > **A (sjperkins):** "The `field_name` coordinate in `correlated_xds` should be used in
   > conjunction with the `field_name` coordinate in `field_and_source_{data_group_name}_xds`
   > to select the appropriate `FIELD_PHASE_CENTER_DIRECTION`. Note `data_group_name` is used
   > to look up the appropriate `field_and_source_xds` node on the DataTree."

9. **Antenna positions** (`s2v:160`): `node_dt["antenna_xds"].ANTENNA_POSITION`.

   > **A (sjperkins):** *(no specific comment — but see Q5: an `np.searchsorted` mapping to
   > `antenna_xds` is required if `ANTENNA_POSITION` is fed into kernels.)*

10. **Polarization** (`s2v:164–169`): linear/circular inferred by set-membership on
    `ds.polarization.values`. *Mixed / other pol frames?*

    > **A (sjperkins):** "yes." *(confirming the set-membership inference on
    > `ds.polarization.values`.)*

11. **UVW frame** — we filter a `FrameConversionWarning` (ITRF → fk5 from
    `FIELD_PHASE_CENTER_DIRECTION`). *Any sign-convention / w-term implications we
    should be acting on rather than silencing?*

    > **A (sjperkins):** "ITRF is not a valid UVW frame, but some CASA tasks will set it (from
    > imperfect memory it's a celestial frame). I don't think this has sign-convention or
    > w-term implications."

12. **Load strategy** — `xr.open_datatree(..., chunks=None)` (eager) then slice
    per `(time-chunk, chan-chunk)` and hand each piece to a Ray task. *Reasonable
    for large MSv4, or should we be using a chunked/lazy read?*

    > **A (sjperkins):** "Yes, this is the strategy. However, for xarray-kat prefer
    > `xr.open_datatree(chunked_array_type="xarray-kat", chunks={})` — this internally sets up
    > chunkwise loading of arrays that share a chunking schema (VISIBILITIES, WEIGHTS, FLAGS).
    > Important because these values are computed dependent on each other; computing them
    > chunkwise reduces the need to fetch data multiple times (due to caching)."

---

## Already validated (so you can skip these)

- imager vs legacy `init`+`grid`, MFS dirty: **natural** and **uniform** (`robustness=0.0`)
  weighting both match to ~1e-14/1e-15 on real data.
- A fully-flagged band is dropped cleanly end to end (no node, no Ray task, no NaN).
- Known limitation, already recorded: the shared Google-Drive test MS currently has an
  all-zero `DATA` column, so the equivalence test is divide-by-zero-safe but vacuous
  in CI until we populate it — see `docs/look-ahead.md §1`.

## Pointers

- Design / rationale: `docs/superpowers/specs/2026-06-04-imager-datatree-design.md`
- Import discipline + imager architecture: `.claude/rules/architecture.md` §3, §8
- Test isolation: `.claude/rules/testing-and-ci.md` §1
- Deferred follow-ups: `docs/look-ahead.md`
