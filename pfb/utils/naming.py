import fsspec

def set_output_names(output_filename, product, fits_output_folder=None):
    '''
    Make sure base folders exist and set default naming conventions
    '''
    if '://' in output_filename:
        protocol = output_filename.split('://')[0]
    else:
        protocol = 'file'

    fs = fsspec.filesystem(protocol)
    basedir = fs.expand_path('/'.join(output_filename.split('/')[:-1]))[0]
    if not fs.exists(basedir):
        fs.makedirs(basedir)

    oname = output_filename.split('/')[-1] + f'_{product.upper()}'

    if fits_output_folder is not None:
        # this should be a file system
        fs = fsspec.filesystem('file')
        fbasedir = fs.expand_path(fits_output_folder)[0]
        if not fs.exists(fbasedir):
            fs.makedirs(fbasedir)
        fits_output_folder = fbasedir
    else:
        fits_output_folder = basedir

    return basedir, oname, fits_output_folder
