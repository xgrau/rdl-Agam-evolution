# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division
import os


import re
import zarr
import petl as etl
import pandas
import h5py


title = 'Phase 1 selection data release'


# noinspection PyGlobalUndefined
def init(release_dir):
    """Initialise data resources.

    Parameters
    ----------
    release_dir : string
        Local filesystem path where data from the release are stored.

    """

    # windowed summary data
    ###########

    global ihs_windowed, xpehh_windowed, xpclr_windowed, h12_windowed, dTjD_windowed 
    ws_dir = os.path.join(release_dir, 'windowed')
    
    for metric in ("ihs", "xpehh", "xpclr", "hstats", "dTjD"):

        fn = os.path.join(ws_dir, '{metric}.txt.gz').format(metric=metric)

        if os.path.exists(fn):
            globals()[metric + "_windowed"] = pandas.read_table(fn).set_index(['chrom', 'start', 'stop'])

    # raw data 
    ##########

    global ihs_raw, xpehh_raw, xpclr_raw, hstats_raw, dTjD_raw
    raw_dir = os.path.join(release_dir, 'raw')
    
    for metric in ("ihs", "xpehh", "hstats", "dTjD"):

        if metric == "hstats":
            fn = os.path.join(raw_dir, '{metric}/output/garud_scan.zarr').format(metric=metric)
        else:
            fn = os.path.join(raw_dir, '{metric}/output.zarr').format(metric=metric)

        if os.path.exists(fn):
            globals()[metric + "_raw"] = zarr.open_group(fn, 'r')

    # XPCLR has no zarr output.
    output_dir = os.path.join(raw_dir, 'xpclr', 'output')
    files = os.listdir(output_dir)

    temp = {}
    for f in files:
        mm = re.search("([A-Za-z0-9]+)_(.+vs.+)\.xpclr\.txt\.gz$", f)
        if mm is not None:
            chrom, comp = mm.groups()
            if chrom not in temp:
                temp[chrom] = {}
            temp[chrom][comp] = pandas.read_table(os.path.join(output_dir, f))

    xpclr_raw = temp
