import fsspec
from functools import partial
import pickle

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
    else:
        protocol = 'file'

    fs = fsspec.filesystem(protocol)
    basedir = fs.expand_path('/'.join(output_filename.split('/')[:-1]))[0]
    if not fs.exists(basedir):
        fs.makedirs(basedir)

    oname = output_filename.split('/')[-1] + f'_{product.upper()}'

    opts.output_filename = f'{basedir}/{oname}'

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
                             f'{protocol}', file=log)
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
                             f'{protocol}', file=log)
        log_directory = basedir

    opts.log_directory = log_directory

    return opts, basedir, oname


def xds_from_url(url, columns='ALL', chunks=-1):
    if columns.upper() != 'ALL':
        raise NotImplementedError
    if chunks != -1:
        raise NotImplementedError

    from daskms.fsspec_store import DaskMSStore
    import xarray as xr
    store = DaskMSStore(url)

    # these will only be read in on first value access and won't be chunked
    open_zarr = partial(xr.open_zarr, chunks=None)
    xds = list(map(open_zarr, store.fs.glob(f'{url}/*.zarr')))
    return xds


def xds_from_list(ds_list, columns='ALL', chunks=-1):
    if columns.upper() != 'ALL':
        raise NotImplementedError
    if chunks != -1:
        raise NotImplementedError
    import xarray as xr

    # these will only be read in on first value access and won't be chunked
    open_zarr = partial(xr.open_zarr, chunks=None)
    return list(map(open_zarr, ds_list))


def cache_opts(opts, url, protocol, name='opts.pkl'):
    fs = fsspec.filesystem(protocol)
    if protocol == 'file':
        with fs.open(f'{url}/{name}', 'wb') as f:
            pickle.dump(opts, f)
    elif protocol == 's3':
        s3_client.put_object(Bucket=url,
                             Key=name,
                             Body=pickle.dumps(opts))
    else:
        raise ValueError(f"protocol {protocol} not yet supported")

def get_opts(url, protocol, name='opts.pkl'):
    fs = fsspec.filesystem(protocol)
    if protocol == 'file':
        with fs.open(f'{url}/{name}', 'rb') as f:
            optsp = pickle.load(f)
    elif protocol == 's3':
        import boto3
        s3_client = boto3.client('s3')
        response = s3_client.get_object(Bucket=url,
                                        Key=name)
        pickled_data = response['Body'].read()
        # Deserialize the object with pickle
        optsp = pickle.loads(pickled_data)
    else:
        raise ValueError(f"protocol {protocol} not yet supported")

    return optsp
