# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division
import os
import itertools


import pyfasta
import allel
import petl as etl
import h5py
import pandas
import zarr
import seaborn as sns


title = 'Phase 2 AR1 release'

pop_ids = (
    'AOcol',
    'GHcol',
    'BFcol',
    'CIcol',
    'GNcol',
    'GW',
    'GM',
    'CMgam',
    'GHgam',
    'BFgam',
    'GNgam',
    'GAgam',
    'UGgam',
    'GQgam',
    'FRgam',
    'KE'
)

pop_labels = {
    'AOcol': 'Angola $coluzzii$',
    'BFcol': 'Burkina Faso $coluzzii$',
    'GHcol': 'Ghana $coluzzii$',
    'CIcol': "Côte d'Ivoire $coluzzii$",
    'GNcol': 'Guinea $coluzzii$',
    'GW': 'Guinea-Bissau',
    'GM': 'The Gambia',
    'GNgam': 'Guinea $gambiae$',
    'BFgam': 'Burkina Faso $gambiae$',
    'GHgam': 'Ghana $gambiae$',
    'CMgam': 'Cameroon $gambiae$',
    'UGgam': 'Uganda $gambiae$',
    'GAgam': 'Gabon $gambiae$',
    'GQgam': 'Bioko $gambiae$',
    'FRgam': 'Mayotte $gambiae$',
    'KE': 'Kenya',
    'colony': 'colony',
}

reds = sns.color_palette('Reds', 5)
blues = sns.color_palette('Blues', 4)
greens = sns.color_palette('Greens', 2)
browns = sns.color_palette('YlOrBr', 4)
purples = sns.color_palette('Purples', 2)
greys = sns.color_palette('Greys', 3)
pop_colors = {
    'AOcol': reds[4],
    'GHcol': reds[3],
    'BFcol': reds[2],
    'CIcol': reds[1],
    'GNcol': reds[0],
    'CMgam': blues[3],
    'GHgam': blues[2],
    'BFgam': blues[1],
    'GNgam': blues[0],
    'GW': browns[1],
    'GM': browns[2],
    'GAgam': greens[1],
    'UGgam': greens[0],
    'FRgam': purples[1],
    'GQgam': purples[0],
    'KE': greys[1],
}


# noinspection PyGlobalUndefined
def init(release_dir, load_geneset=False, geneset_attributes=None):
    """Initialise data resources.

    Parameters
    ----------
    release_dir : string
        Local filesystem path where data from the release are stored.
    load_geneset : string
        If True, load geneset into memory.
    geneset_attributes : dict-like
        Attributes to load.

    """

    # reference sequence
    ####################

    global genome_agamp3, genome_agamp4, genome_dir
    genome_dir = os.path.join(release_dir, 'genome')
    genome_agamp3_dir = os.path.join(genome_dir, 'agamP3')
    genome_agamp3_fn = os.path.join(genome_agamp3_dir, 'Anopheles-gambiae-PEST_CHROMOSOMES_AgamP3.fa')
    if os.path.exists(genome_agamp3_fn):
        genome_agamp3 = pyfasta.Fasta(genome_agamp3_fn, key_fn=lambda v: v.split()[0])
    genome_agamp4_dir = os.path.join(genome_dir, 'agamP4')
    genome_agamp4_fn = os.path.join(genome_agamp4_dir, 'Anopheles-gambiae-PEST_CHROMOSOMES_AgamP4.fa')
    if os.path.exists(genome_agamp4_fn):
        genome_agamp4 = pyfasta.Fasta(genome_agamp4_fn, key_fn=lambda v: v.split()[0])

    # genome annotations
    ####################

    global geneset_agamp44_fn, geneset_agamp44, geneset_dir
    geneset_dir = os.path.join(release_dir, 'geneset')
    geneset_agamp44_fn = os.path.join(geneset_dir, 'Anopheles-gambiae-PEST_BASEFEATURES_AgamP4.4.sorted.gff3.gz')
    if load_geneset:
        geneset_agamp44 = allel.FeatureTable.from_gff3(
            geneset_agamp44_fn,
            attributes=geneset_attributes)

    # variant callsets
    ##################

    global callset, callset_pass, callset_pass_biallelic, variation_dir, \
        callset_snpeff_agamp42
    variation_dir = os.path.join(release_dir, 'variation')

    # main callset
    callset_h5_fn = os.path.join(variation_dir, 'main', 'hdf5', 'all', 'ag1000g.phase2.ar1.h5')
    callset_lite_h5_fn = os.path.join(variation_dir, 'main', 'hdf5', 'lite', 'ag1000g.phase2.ar1.lite.h5')
    callset_zarr_fn = os.path.join(variation_dir, 'main', 'zarr2', 'ag1000g.phase2.ar1')

    # preference: zarr > hdf5 > hdf5 (lite)
    if os.path.exists(callset_zarr_fn):
        callset = zarr.open_group(callset_zarr_fn, mode='r')
    elif os.path.exists(callset_h5_fn):
        callset = h5py.File(callset_h5_fn, mode='r')
    elif os.path.exists(callset_lite_h5_fn):
        callset = h5py.File(callset_lite_h5_fn, mode='r')

    # main callset, PASS variants only
    callset_pass_h5_fn = os.path.join(variation_dir, 'main', 'hdf5', 'pass', 'ag1000g.phase2.ar1.pass.h5')
    callset_pass_lite_h5_fn = os.path.join(variation_dir, 'main', 'hdf5', 'lite', 'ag1000g.phase2.ar1.pass.lite.h5')
    callset_pass_zarr_fn = os.path.join(variation_dir, 'main', 'zarr2', 'ag1000g.phase2.ar1.pass')

    # preference: zarr > hdf5 > hdf5 (lite)
    if os.path.exists(callset_pass_zarr_fn):
        callset_pass = zarr.open_group(callset_pass_zarr_fn, mode='r')
    elif os.path.exists(callset_pass_h5_fn):
        callset_pass = h5py.File(callset_pass_h5_fn, mode='r')
    elif os.path.exists(callset_pass_lite_h5_fn):
        callset_pass = h5py.File(callset_pass_lite_h5_fn, mode='r')

    # main callset, PASS biallelic variants only
    callset_pass_biallelic_h5_fn = os.path.join(variation_dir, 'main', 'hdf5', 'biallelic', 'ag1000g.phase2.ar1.pass.biallelic.h5')
    callset_pass_biallelic_lite_h5_fn = os.path.join(variation_dir, 'main', 'hdf5', 'lite', 'ag1000g.phase2.ar1.pass.biallelic.lite.h5')
    callset_pass_biallelic_zarr_fn = os.path.join(variation_dir, 'main', 'zarr2', 'ag1000g.phase2.ar1.pass.biallelic')

    # preference: zarr > hdf5 > hdf5 (lite)
    if os.path.exists(callset_pass_biallelic_zarr_fn):
        callset_pass_biallelic = zarr.open_group(callset_pass_biallelic_zarr_fn, mode='r')
    elif os.path.exists(callset_pass_biallelic_h5_fn):
        callset_pass_biallelic = h5py.File(callset_pass_biallelic_h5_fn, mode='r')
    elif os.path.exists(callset_pass_biallelic_lite_h5_fn):
        callset_pass_biallelic = h5py.File(callset_pass_biallelic_lite_h5_fn, mode='r')

    # SNPEFF annotations
    callset_snpeff_agamp42_h5_fn_template = os.path.join(
        variation_dir, 'main', 'hdf5', 'all_snpeff',
        'ag1000g.phase2.ar1.snpeff.AgamP4.2.{chrom}.h5'
    )
    # work around broken link file
    callset_snpeff_agamp42 = dict()
    for chrom in '2L', '2R', '3L', '3R', 'X':
        fn = callset_snpeff_agamp42_h5_fn_template.format(chrom=chrom)
        if os.path.exists(fn):
            callset_snpeff_agamp42[chrom] = h5py.File(fn, mode='r')[chrom]

    # accessibility
    ###############

    global accessibility, accessibility_dir
    accessibility_dir = os.path.join(release_dir, 'accessibility')
    accessibility_fn = os.path.join(accessibility_dir, 'accessibility.h5')
    if os.path.exists(accessibility_fn):
        accessibility = h5py.File(accessibility_fn, mode='r')

    # sample metadata
    #################

    global tbl_samples, lkp_samples, sample_ids, df_samples, samples_dir
    samples_dir = os.path.join(release_dir, 'samples')
    samples_fn = os.path.join(samples_dir, 'samples.meta.txt')
    if os.path.exists(samples_fn):
        tbl_samples = (
            etl
            .fromtsv(samples_fn)
            .convert(('year', 'n_sequences'), int)
            .convert(('mean_coverage',), float)
        )
        lkp_samples = tbl_samples.recordlookupone('ox_code')
        sample_ids = tbl_samples.values('ox_code').list()
        df_samples = pandas.read_csv(samples_fn, sep='\t', index_col='ox_code')

    # extras
    ########

    global allele_counts
    extras_dir = os.path.join(release_dir, 'extras')

    # allele counts
    allele_counts_fn = os.path.join(extras_dir, 'allele_counts.h5')
    if os.path.exists(allele_counts_fn):
        allele_counts = h5py.File(allele_counts_fn, mode='r')

    # haplotypes
    ############

    global haplotypes_dir, callset_phased, tbl_haplotypes, df_haplotypes, lkp_haplotypes
    haplotypes_dir = os.path.join(release_dir, 'haplotypes')

    # no HDF5 link file, load up as dict for now
    callset_phased_hdf5_fn_template = os.path.join(haplotypes_dir, 'main', 'hdf5',
                                                   'ag1000g.phase2.ar1.haplotypes.{chrom}.h5')
    callset_phased = dict()
    for chrom in '2L', '2R', '3L', '3R', 'X':
        fn = callset_phased_hdf5_fn_template.format(chrom=chrom)
        if os.path.exists(fn):
            callset_phased[chrom] = h5py.File(fn, mode='r')[chrom]

    # no haplotypes file, create here for now
    # TODO source this from file Nick has created
    if '3R' in callset_phased:
        phased_samples = callset_phased['3R']['samples'][:].astype('U')
        haplotype_labels = list(itertools.chain(*[[s + 'a', s + 'b'] for s in phased_samples]))
        tbl_haplotypes = (
            etl
            .empty()
            .addcolumn('label', haplotype_labels)
            .addrownumbers(start=0)
            .rename('row', 'index')
            .addfield('ox_code', lambda row: row.label[:-1])
            .hashleftjoin(tbl_samples, key='ox_code')
            .addfield('label_aug', lambda row: '%s [%s, %s, %s, %s]' % (row.label, row.country, row.region, row.m_s, row.sex))
        )
        lkp_haplotypes = tbl_haplotypes.recordlookupone('label')
        df_haplotypes = tbl_haplotypes.todataframe(index='index')

