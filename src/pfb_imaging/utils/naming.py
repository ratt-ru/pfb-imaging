import fsspec


def set_output_names(
    output_filename: str, product: str, fits_output_folder: str | None = None, log_directory: str | None = None
):
    """
    Make sure base folders exist and set default naming conventions
    """
    if "://" in output_filename:
        protocol = output_filename.split("://")[0]
        prefix = f"{protocol}://"
    else:
        protocol = "file"
        prefix = ""

    fs = fsspec.filesystem(protocol)
    basedir = fs.expand_path("/".join(output_filename.split("/")[:-1]))[0]
    if not fs.exists(basedir):
        fs.makedirs(basedir)

    oname = output_filename.split("/")[-1] + f"_{product.upper()}"

    output_filename = f"{prefix}{basedir}/{oname}"

    if fits_output_folder is not None:
        # this should be a file system
        fs = fsspec.filesystem("file")
        fbasedir = fs.expand_path(fits_output_folder)[0]
        if not fs.exists(fbasedir):
            fs.makedirs(fbasedir)
        fits_output_folder = fbasedir
    else:
        if protocol != "file":
            raise ValueError(f"You must provide a separate fits-output-folder when output protocol is {protocol}")
        fits_output_folder = f"{basedir}/fits"
        if not fs.exists(fits_output_folder):
            fs.makedirs(fits_output_folder)

    if log_directory is not None:
        # this should be a file system
        fs = fsspec.filesystem("file")
        lbasedir = fs.expand_path(log_directory)[0]
        if not fs.exists(lbasedir):
            fs.makedirs(lbasedir)
        log_directory = lbasedir
    else:
        if protocol != "file":
            raise ValueError(f"You must provide a separate log-directory when output protocol is {protocol}")
        log_directory = f"{basedir}/logs"
        if not fs.exists(log_directory):
            fs.makedirs(log_directory)

    return output_filename, fits_output_folder, log_directory, oname


def uri_and_fs(path):
    """Filesystem and absolute, protocol-qualified URI for a path or URL.

    Replaces `daskms.fsspec_store.DaskMSStore` for the store-lifecycle calls
    (`exists`/`rm`/`makedirs`/`.url`) the pipeline actually makes. DaskMSStore
    predates the MSv4 migration; for a plain store -- no `path::SUBTABLE`
    syntax -- its `.url` is exactly `fs.unstrip_protocol(fs._strip_protocol(p))`,
    verified equal for relative paths, trailing slashes, `PurePath`, explicit
    `file://` and `s3://`. Dropping it removes a dask-ms (and therefore
    python-casacore) dependency from paths that only ever needed fsspec.

    Two behavioural differences, both deliberate:

    * DaskMSStore looked `storage_options` up in dask-ms's own config by URL
      prefix. Nothing in pfb-imaging ever set that, and fsspec's own
      configuration still applies, so remote credentials are now configured the
      ordinary fsspec way.
    * `::` is rejected rather than interpreted. dask-ms reads `path::SUBTABLE`
      as CASA subtable syntax; fsspec reads `a::b` as a chained URL and would
      resolve `/data/weird::name.ms` to `/data/weird` -- a different path that
      exists and is writable. Neither reading is ours (MSv4 subtables are
      DataTree nodes, not paths), and silently writing to the wrong store is
      much worse than a refusal. Chained URLs never worked here either: through
      DaskMSStore, `simplecache::file:///x` resolved to a literal directory of
      that name under the cwd.

    Args:
        path: Path, URL or `PurePath`. Relative paths are made absolute.

    Returns:
        `(fs, uri)`: the `AbstractFileSystem` and the normalised URI. Pass the
        URI straight back to the filesystem's methods, or to zarr/xarray.

    Raises:
        ValueError: If the path contains `::`.
    """
    path = str(path)
    if "::" in path:
        raise ValueError(
            f"'::' is not supported in a store path ({path!r}). fsspec reads it as chained-URL "
            "syntax and would resolve this to a different location; rename the path."
        )
    fs, root = fsspec.core.url_to_fs(path)
    return fs, fs.unstrip_protocol(root)


def glob_uris(pattern):
    """Expand a glob pattern to absolute, protocol-qualified URIs.

    Args:
        pattern: Glob pattern, with or without a protocol. A trailing slash is
            stripped, so `foo.ms/` and `foo.ms` glob alike.

    Returns:
        Matching URIs, empty when nothing matches. Callers report the "no MS
        at ..." error themselves, because they know what they were looking for.
    """
    pattern = str(pattern).rstrip("/")
    fs, _ = uri_and_fs(pattern)
    return [fs.unstrip_protocol(p) for p in fs.glob(pattern)]
