# -*- coding: utf-8 -*-

"""Utility functions for test datasets."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import hashlib
import os
import os.path as op

try:
    from requests import get
except ImportError:
    get = None

from .logging import info


#------------------------------------------------------------------------------
# Utility functions
#------------------------------------------------------------------------------

_BASE_URL = {
    'cortexlab': 'http://phy.cortexlab.net/data/',
    'github': 'https://raw.githubusercontent.com/kwikteam/phy-data/master/',
}


def md5(path, blocksize=2 ** 20):
    """Compute the checksum of a file."""
    m = hashlib.md5()
    with open(path, 'rb') as f:
        while True:
            buf = f.read(blocksize)
            if not buf:
                break
            m.update(buf)
    return m.hexdigest()


def download_file(url, output=None, checksum=None):
    """Download a binary file from an URL."""
    if output is None:
        output = url.split('/')[-1]
    if op.exists(output):
        info("The file {} already exists: skipping.".format(output))
        return
    if not get:
        raise ImportError("Please install the requests package.")
    r = get(url, stream=True)
    info("Downloading {0}...".format(url))
    with open(output, 'wb') as f:
        for chunk in r.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)
                f.flush()
    if checksum is not None:
        if md5(output) != checksum:
            raise RuntimeError("The checksum of the downloaded file doesn't "
                               "match the provided checksum.")


def _download(url):
    if not get:
        raise ImportError("Please install the requests package.")
    return get(url).text


def download_test_data(name, output_dir=None, base='cortexlab'):
    """Download a test dataset.

    Parameters
    ----------

    name : str
        Name of the test dataset to download.
    output_dir : str
        The directory where to save the file.
    base : str
        The id of the base URL. Can be `'cortexlab'` or `'github'`.

    """
    if output_dir is None:
        output_dir = name
    if not output_dir.endswith('/'):
        output_dir = output_dir + '/'
    output_dir = op.realpath(op.dirname(output_dir))
    if not op.exists(output_dir):
        os.mkdir(output_dir)
    for ext in ('.kwik', '.kwx', '.raw.kwd'):
        url = _BASE_URL[base] + name + ext
        output = op.join(output_dir, name + ext)
        url_checksum = _BASE_URL[base] + name + ext + '.md5'
        checksum = _download(url_checksum).split(' ')[0]
        download_file(url, output=output, checksum=checksum)
