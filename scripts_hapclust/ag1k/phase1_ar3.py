# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division
import os


import pyfasta
import allel
import seaborn as sns
import petl as etl
import h5py
import pandas


title = 'Phase 1 AR3 release'

pop_ids = 'AOM', 'BFM', 'GWA', 'GNS', 'BFS', 'CMS', 'GAS', 'UGS', 'KES'

pop_labels = {
    'AOM': 'AO $coluzzii$',
    'BFM': 'BF $coluzzii$',
    'GWA': 'GW',
    'GNS': 'GN $gambiae$',
    'BFS': 'BF $gambiae$',
    'CMS': 'CM $gambiae$',
    'UGS': 'UG $gambiae$',
    'GAS': 'GA $gambiae$',
    'KES': 'KE',
    'colony': 'colony',
}

pop_colors = {
    'AOM': sns.color_palette('YlOrBr', 5)[4],
    'BFM': sns.color_palette('Reds', 3)[1],
    'GWA': sns.color_palette('YlOrBr', 5)[1],
    'GNS': sns.color_palette('Blues', 3)[0],
    'BFS': sns.color_palette('Blues', 3)[1],
    'CMS': sns.color_palette('Blues', 3)[2],
    'UGS': sns.color_palette('Greens', 2)[0],
    'GAS': sns.color_palette('Greens', 2)[1],
    'KES': sns.color_palette('Greys', 5)[2],
    'colony': sns.color_palette('Greys', 5)[-1]
}
# convert to hex notation for ease of use elsewhere
for p in pop_colors:
    h = '#%02x%02x%02x' % tuple(int(255*c) for c in pop_colors[p])

# chromatin
_data_chromatin = b"""CHX     chro    X       20009764        24393108
CH2R    chro    2R      58984778        61545105
CH2L    chro    2L      1       2431617
PEU2L   chro    2L      2487770 5042389
IH2L    chro    2L      5078962 5788875
IH3R    chro    3R      38988757        41860198
CH3R    chro    3R      52161877        53200684
CH3L    chro    3L      1       1815119
PEU3L   chro    3L      1896830 4235209
IH3L    chro    3L      4264713 5031692
"""
tbl_chromatin = (
    etl
    .fromtext(etl.MemorySource(_data_chromatin))
    .split('lines', '\s+', ['name', 'type', 'chrom', 'start', 'stop'])
    .convert(('start', 'stop'), int)
    .cutout('type')
)

# genome regions
region_X_speciation = 'X-speciation', 'X', 15000000, 24000000
region_X_free = 'X-free', 'X', 1, 14000000
region_3L_free = '3L-free', '3L', 15000000, 41000000
region_3R_free = '3R-free', '3R', 1, 37000000


# noinspection PyGlobalUndefined
def init(release_dir, load_geneset=False):
    """Initialise data resources.

    Parameters
    ----------
    release_dir : string
        Local filesystem path where data from the release are stored.
    load_geneset : string
        If True, load geneset into memory.

    """

    # reference sequence
    ####################

    global genome_fn, genome
    genome_dir = os.path.join(release_dir, 'genome')
    genome_fn = os.path.join(genome_dir, 'Anopheles-gambiae-PEST_CHROMOSOMES_AgamP3.fa')
    if os.path.exists(genome_fn):
        genome = pyfasta.Fasta(genome_fn)

    # genome annotations
    ####################

    global geneset_agamp42_fn, geneset_agamp42
    geneset_dir = os.path.join(release_dir, 'geneset')
    geneset_agamp42_fn = os.path.join(
        geneset_dir,
        'Anopheles-gambiae-PEST_BASEFEATURES_AgamP4.2.sorted.gff3.gz')
    if os.path.exists(geneset_agamp42_fn) and load_geneset:
        geneset_agamp42 = allel.FeatureTable.from_gff3(geneset_agamp42_fn)

    # variant callsets
    ##################

    global callset, callset_pass
    variation_dir = os.path.join(release_dir, 'variation')

    # main callset
    callset_h5_fn = os.path.join(variation_dir, 'main', 'hdf5', 'ag1000g.phase1.ar3.h5')
    if os.path.exists(callset_h5_fn):
        callset = h5py.File(callset_h5_fn, mode='r')

    # main callset, PASS variants only
    callset_pass_h5_fn = os.path.join(variation_dir, 'main', 'hdf5', 'ag1000g.phase1.ar3.pass.h5')
    if os.path.exists(callset_pass_h5_fn):
        callset_pass = h5py.File(callset_pass_h5_fn, mode='r')

    # accessibility
    ###############

    global accessibility
    accessibility_dir = os.path.join(release_dir, 'accessibility')
    accessibility_fn = os.path.join(accessibility_dir, 'accessibility.h5')
    if os.path.exists(accessibility_fn):
        accessibility = h5py.File(accessibility_fn, mode='r')

    # sample metadata
    #################

    global samples_fn, tbl_samples, lkp_samples, sample_ids, df_samples
    samples_dir = os.path.join(release_dir, 'samples')
    samples_fn = os.path.join(samples_dir, 'samples.all.txt')
    if os.path.exists(samples_fn):
        tbl_samples = (
            etl
            .fromtsv(samples_fn)
            .convert(('index', 'year', 'n_sequences', 'kt_2la', 'kt_2rb'), int)
            .convert(('mean_coverage', 'latitude', 'longitude') + tuple(range(20, 36)), float)
        )
        lkp_samples = tbl_samples.recordlookupone('ox_code')
        sample_ids = tbl_samples.values('ox_code').list()
        df_samples = pandas.read_csv(samples_fn, sep='\t', index_col='index')

    # extras
    ########

    global allele_counts, allele_counts_gq10, outgroup_alleles, outgroup_allele_counts, \
        outgroup_species
    extras_dir = os.path.join(release_dir, 'extras')

    # allele counts
    allele_counts_fn = os.path.join(extras_dir, 'allele_counts.h5')
    if os.path.exists(allele_counts_fn):
        allele_counts = h5py.File(allele_counts_fn, mode='r')
    allele_counts_gq10_fn = os.path.join(extras_dir, 'allele_counts.gq10.h5')
    if os.path.exists(allele_counts_gq10_fn):
        allele_counts_gq10 = h5py.File(allele_counts_gq10_fn, mode='r')

    # outgroup data
    outgroup_species = 'arab', 'meru', 'mela', 'quad', 'epir', 'chri'
    outgroup_alleles_fn = os.path.join(extras_dir, 'outgroup_alleles.h5')
    if os.path.exists(outgroup_alleles_fn):
        outgroup_alleles = h5py.File(outgroup_alleles_fn, mode='r')
    outgroup_allele_counts_fn = os.path.join(extras_dir, 'outgroup_allele_counts.h5')
    if os.path.exists(outgroup_allele_counts_fn):
        outgroup_allele_counts = h5py.File(outgroup_allele_counts_fn, mode='r')
