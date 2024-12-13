import fsspec
import s3fs
from functools import partial
import pickle
import xarray as xr
import concurrent.futures as cf

def set_output_names(opts):
    '''
    Make sure base folders exist and set default naming conventions
    '''
    output_filename = opts.output_filename
    product = opts.product
    fits_output_folder = opts.fits_output_folder
    log_directory = opts.log_directory

    if '://' in output_filename:
        protocol = output_filename.split('://')[0]
        prefix = f'{protocol}://'
    else:
        protocol = 'file'
        prefix = ''

    fs = fsspec.filesystem(protocol)
    basedir = fs.expand_path('/'.join(output_filename.split('/')[:-1]))[0]
    if not fs.exists(basedir):
        fs.makedirs(basedir)

    oname = output_filename.split('/')[-1] + f'_{product.upper()}'

    opts.output_filename = f'{prefix}{basedir}/{oname}'

    if fits_output_folder is not None:
        # this should be a file system
        fs = fsspec.filesystem('file')
        fbasedir = fs.expand_path(fits_output_folder)[0]
        if not fs.exists(fbasedir):
            fs.makedirs(fbasedir)
        fits_output_folder = fbasedir
    else:
        if protocol != 'file':
            raise ValueError('You must provide a separate fits-output-'
                             'folder when output protocol is '
                             f'{protocol}')
        fits_output_folder = basedir

    opts.fits_output_folder = fits_output_folder

    if log_directory is not None:
        # this should be a file system
        fs = fsspec.filesystem('file')
        lbasedir = fs.expand_path(log_directory)[0]
        if not fs.exists(lbasedir):
            fs.makedirs(lbasedir)
        log_directory = lbasedir
    else:
        if protocol != 'file':
            raise ValueError('You must provide a separate log-'
                             'directory when output protocol is '
                             f'{protocol}')
        log_directory = basedir

    opts.log_directory = log_directory

    return opts, basedir, oname


def xds_from_url(url, columns='ALL', chunks=-1):
    '''
    Returns a lazy view of all datasets contained in url
    as well as a list of full paths to each one
    '''
    if columns.upper() != 'ALL':
        raise NotImplementedError
    if chunks != -1:
        raise NotImplementedError

    url = url.rstrip('/')
    if '://' in url:
        protocol = url.split('://')[0]
        prefix = f'{protocol}://'
    else:
        protocol = 'file'
        prefix = ''
    fs = fsspec.filesystem(protocol)
    ds_list = fs.glob(f'{url}/*.zarr')
    ds_list = list(map(lambda x: prefix + x, ds_list))
    # these will only be read in on first value access and won't be chunked
    open_zarr = partial(xr.open_zarr, chunks=None)
    xds = list(map(open_zarr, ds_list))

    if not len(xds):
        raise ValueError(f'Nothing found at {url}')
    return xds, ds_list


def read_var(ds, var):
    '''
    Access var to force loading it into memory
    '''
    val = getattr(ds, var).values
    return 1


def xds_from_list(ds_list, drop_vars=None, drop_all_but=None,
                  chunks=-1, nthreads=1, order_freq=True):
    '''
    Reads a list of datasets into memory in parallel.
    Use drop_vars to drop vars that should not be read into memory.
    Shoudl return a list that is ordered in frequency
    '''
    if chunks != -1:
        raise NotImplementedError

    # these will only be read in on first value access and won't be chunked
    open_zarr = partial(xr.open_zarr, chunks=None)
    xds = list(map(open_zarr, ds_list))
    if not len(xds):
        raise ValueError(f'Nothing found at {ds_list[0]}')

    if drop_vars is not None:
        if not isinstance(drop_vars, list):
            drop_vars = list(drop_vars)
        for i, ds in enumerate(xds):
            xds[i] = ds.drop_vars(drop_vars, errors="ignore")

    if drop_all_but is not None:
        if not isinstance(drop_all_but, list):
            drop_all_but = [drop_all_but]
        for i, ds in enumerate(xds):
            drop_vars = [var for var in ds.data_vars]
            for var in drop_all_but:
                if var in drop_vars:
                    drop_vars.pop(drop_vars.index(var))
            xds[i] = ds.drop_vars(drop_vars, errors="ignore")

    futures = []
    with cf.ThreadPoolExecutor(max_workers=nthreads) as executor:
        for ds in xds:
            for var in ds.data_vars:
                futures.append(executor.submit(read_var, ds, var))
    cf.wait(futures)

    # make sure list is ordered correctly (increasing frequency)
    # TODO - reorder if not
    if len(xds) > 1 and order_freq:
        freq_out = [ds.freq_out for ds in xds]
        freq_min = freq_out[0]
        for f in freq_out[1:]:
            if freq_min > f:
                raise ValueError('xds list is not ordered in frequency')
            freq_min = f
    return xds


def cache_opts(opts, url, protocol, name='opts.pkl'):
    fs = fsspec.filesystem(protocol)
    if protocol == 'file':
        with fs.open(f'{url}/{name}', 'wb') as f:
            pickle.dump(opts, f)
    elif protocol == 's3':
        s3_client = s3fs.S3FileSystem(anon=False)
        bucket = url.split('://')[1]
        s3_path = f"{bucket}/{name}"
        with s3_client.open(s3_path, 'wb') as f:
            pickle.dump(opts, f)
    else:
        raise ValueError(f"protocol {protocol} not yet supported")

def get_opts(url, protocol, name='opts.pkl'):
    fs = fsspec.filesystem(protocol)
    if protocol == 'file':
        with fs.open(f'{url}/{name}', 'rb') as f:
            optsp = pickle.load(f)
    elif protocol == 's3':
        s3_client = s3fs.S3FileSystem(anon=False)
        bucket = url.split('://')[1]
        s3_path = f"{bucket}/{name}"
        with s3_client.open(s3_path, 'rb') as f:
            optsp = pickle.load(f)
    else:
        raise ValueError(f"protocol {protocol} not yet supported")

    return optsp
