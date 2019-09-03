# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division
import os


import zarr
import petl as etl
import pandas
import h5py


title = 'Phase 1 AR3.1 release'


# noinspection PyGlobalUndefined
def init(release_dir):
    """Initialise data resources.

    Parameters
    ----------
    release_dir : string
        Local filesystem path where data from the release are stored.

    """

    # variation
    ###########

    global callset, callset_pass
    variation_dir = os.path.join(release_dir, 'variation')

    # main callset
    callset_zarr_fn = os.path.join(variation_dir, 'main', 'zarr2', 'ag1000g.phase1.ar3')
    if os.path.exists(callset_zarr_fn):
        callset = zarr.open_group(callset_zarr_fn, mode='r')

    # main callset, PASS variants only
    callset_pass_zarr_fn = os.path.join(variation_dir, 'main', 'zarr2', 'ag1000g.phase1.ar3.pass')
    if os.path.exists(callset_pass_zarr_fn):
        callset_pass = zarr.open_group(callset_pass_zarr_fn, mode='r')

    # haplotypes
    ############

    global callset_phased, tbl_haplotypes, lkp_haplotypes, df_haplotypes
    haplotypes_dir = os.path.join(release_dir, 'haplotypes')

    # try HDF5 first
    callset_phased_h5_fn = os.path.join(haplotypes_dir, 'main', 'hdf5',
                                        'ag1000g.phase1.ar3.1.haplotypes.h5')
    if os.path.exists(callset_phased_h5_fn):
        callset_phased = h5py.File(callset_phased_h5_fn, mode='r')

    # prefer Zarr if available
    # N.B., the Zarr data is not consistent with HDF5 or shapeit outputs,
    # it is based on a previous phasing run.
    #
    #callset_phased_zarr_fn = os.path.join(haplotypes_dir, 'main', 'zarr2',
    #                                      'ag1000g.phase1.ar3.1.haplotypes')
    #if os.path.exists(callset_phased_zarr_fn):
    #    callset_phased = zarr.open_group(callset_phased_zarr_fn, mode='r')

    # haplotypes metadata
    haplotypes_fn = os.path.join(haplotypes_dir, 'haplotypes.meta.txt')
    if os.path.exists(haplotypes_fn):
        tbl_haplotypes = (
            etl
            .fromtsv(haplotypes_fn)
            .convert(('index', 'kt_2la', 'kt_2rb'), int)
        )
        lkp_haplotypes = tbl_haplotypes.recordlookupone('label')
        df_haplotypes = pandas.read_csv(haplotypes_fn, sep='\t', index_col='index')
