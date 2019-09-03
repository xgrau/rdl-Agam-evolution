# -*- coding: utf-8 -*-
"""
General purpose utility functions.

"""
from __future__ import absolute_import, print_function, division


# standard library imports
import contextlib
import sys
import datetime


# third party library imports
import humanize
import pandas
import petl as etl
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns


_slog_indent = -2


def log(*msg):
    """Simple logging function that flushes immediately to stdout."""
    s = ' '.join(map(str, msg))
    print(s, file=sys.stdout)
    sys.stdout.flush()


@contextlib.contextmanager
def timer(*msg):
    before = datetime.datetime.now()
    try:
        yield
    except:
        after = datetime.datetime.now()
        elapsed = (after - before).total_seconds()
        done = 'errored after %s' % humanize.naturaldelta(elapsed)
        if not msg:
            msg = done
        else:
            msg = ', '.join(map(str, msg)) + ', ' + done
        print(msg, file=sys.stderr)
        sys.stderr.flush()
        raise
    else:
        after = datetime.datetime.now()
        elapsed = (after - before).total_seconds()
        done = 'done in %s' % humanize.naturaldelta(elapsed)
        if not msg:
            msg = done
        else:
            msg = ', '.join(map(str, msg)) + ', ' + done
        print(msg, file=sys.stdout)
        sys.stdout.flush()


@contextlib.contextmanager
def section(*title):
    global _slog_indent
    before = datetime.datetime.now()
    _slog_indent += 2
    prefix = (' ' * _slog_indent) + '[' + ', '.join(map(str, title)) + '] '

    def slog(*msg, file=sys.stdout):
        print(prefix + ' '.join(map(str, msg)), file=file)
        file.flush()

    slog('begin')

    try:
        yield slog

    except:
        after = datetime.datetime.now()
        elapsed = (after - before).total_seconds()
        msg = 'errored after %s' % humanize.naturaldelta(elapsed)
        slog(msg, file=sys.stderr)
        _slog_indent -= 2
        raise

    else:
        after = datetime.datetime.now()
        elapsed = (after - before).total_seconds()
        msg = 'done in %s' % humanize.naturaldelta(elapsed)
        slog(msg, file=sys.stdout)
        _slog_indent -= 2


def _h5ls(h5o, currentdepth, maxdepth, maxitems, prefix):
    if maxdepth is not None and currentdepth == maxdepth:
        return
    for i, k in enumerate(h5o.keys()):
        path = prefix + '/' + k
        if maxitems is not None and i == maxitems:
            print(prefix + '/...')
            break
        v = h5o[k]
        print(path + ' : ' + repr(v))
        if hasattr(v, 'keys'):
            _h5ls(v, currentdepth+1, maxdepth=maxdepth, maxitems=maxitems, prefix=path)


def h5ls(h5o, maxdepth=None, maxitems=None):
    """Obtain a recursive listing of the contents of an HDF5 file or group."""
    _h5ls(h5o, 0, maxdepth=maxdepth, maxitems=maxitems, prefix='')


def fig_linear_genome(plotf, genome, chromosomes=('2R', '2L', '3R', '3L', 'X'),
                      fig=None, bottom=0, height=1, width_factor=1.08, chrom_pad=0.035,
                      clip_patch_kwargs=None, **kwargs):
    """Utility function to make a linear genome figure.

    Parameters
    ----------
    plotf : function
        Function to plot a single chromosome. Must accept 'chrom' and 'ax' arguments and also
        flexible **kwargs.
    genome : pyfasta.Fasta
        Reference sequence. Used to compute genome and chromosome sizes.
    chromosomes : tuple of strings, optional
        Chromosomes to plot.
    fig : figure, optional
        Figure to plot on. If not provided, a new figure will be created.
    bottom : float, optional
        Figure coordinate to position bottom of axes.
    height : float, optional
        Figure height to use for axes.
    width_factor : float, optional
        Used to scale width of each chromosome subplot.
    chrom_pad : float, optional
        Used to set padding between chromosomes.
    clip_patch_kwargs : dict-like, optional
        Arguments for the clip path if used.
    **kwargs
        Passed through to `plotf`.

    Returns
    -------
    axs : dict
        A dictionary mapping chromosome names onto axes objects.

    """

    from matplotlib.path import Path

    # compute assembled genome size
    genome_size = sum(len(genome[chrom]) for chrom in chromosomes)

    # setup figure
    if fig is None:
        fig = plt.figure(figsize=(8, 1))

    # setup clip patch
    if clip_patch_kwargs is None:
        clip_patch_kwargs = dict()
    clip_patch_kwargs.setdefault('edgecolor', 'k')
    clip_patch_kwargs.setdefault('facecolor', 'none')
    clip_patch_kwargs.setdefault('lw', 1)

    # setup axes
    left = 0
    axs = dict()

    for chrom in chromosomes:

        # calculate width needed for this chrom
        width = len(genome[chrom]) / (genome_size * width_factor)

        # create axes
        ax = fig.add_axes([left, bottom, width, height])
        ax.set_facecolor((1, 1, 1, 0))
        axs[chrom] = ax

        # construct clip path
        if chrom in {'2R', '3R'}:
            verts = [(0.01, 0.02), (0.9, 0.02), (1.01, 0.3), (1.01, 0.7), (0.9, .98), (0.01, .98), (0.01, 0.02)]
            codes = [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY]
        elif chrom == "X":
            verts = [(0.01, 0.02), (0.9, 0.02), (0.99, 0.3), (0.99, 0.7), (0.9, .98), (0.01, .98), (0.01, 0.02)]
            codes = [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY]
        else:
            verts = [(0.1, 0.02), (.99, 0.02), (.99, .98), (.1, .98), (-0.01, .7), (-0.01, .3), (0.1, 0.02)]
            codes = [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY]
        path = Path(verts, codes)
        clip_patch = mpl.patches.PathPatch(path, transform=ax.transAxes, **clip_patch_kwargs)

        # do the plotting
        plotf(chrom=chrom, ax=ax, clip_patch=clip_patch, **kwargs)

        # increment left coordinate
        left += len(genome[chrom]) / (genome_size * width_factor)
        if chrom in {'2L', '3L'}:
            left += chrom_pad

    return axs


def geneset_to_pandas(geneset):
    """Life is a bit easier when a geneset is a pandas DataFrame."""
    items = []
    for n in geneset.dtype.names:
        v = geneset[n]
        # convert bytes columns to unicode (which pandas then converts to object)
        if v.dtype.kind == 'S':
            v = v.astype('U')
        items.append((n, v))
    return pandas.DataFrame.from_items(items)


class SeqFeature(object):
    """Genomic sequence feature, with utilities for mapping between coordinate systems.

    Parameters
    ----------
    seqid : string
        Chromosome or contig.
    start : int
        Start coordinate, 1-based.
    end : int
        End coordinate, 1-based, end-inclusive.

    """

    def __init__(self, seqid, start, end, strand=None, genome=None, label=None):
        self.seqid = seqid
        self.start = start
        self.end = end
        self.strand = strand
        self.genome = genome
        self.label = label

    @property
    def loc0(self):
        """A zero-based stop-exclusive slice."""
        return slice(self.start - 1, self.end)

    @property
    def query_str(self):
        """A pandas-style query string."""
        return "(seqid == %r) & (start >= %s) & (end <= %s)" % (self.seqid, self.start, self.end)

    @property
    def region_str(self):
        """A samtools-style region string."""
        return "%s:%s-%s" % (self.seqid, self.start, self.end)

    @property
    def seq(self):
        """The reference sequence."""
        return self.genome[self.seqid][self.loc0]

    def __len__(self):
        # include start and end positions in length
        return self.end - self.start + 1

    def __iter__(self):
        yield self.seqid
        yield self.start
        yield self.end

    def __repr__(self):
        r = '<%s' % type(self).__name__
        if self.label:
            r += ' %r' % self.label
        r += ' ' + self.region_str
        if self.strand:
            r += ' (%s)' % self.strand
        r += '>'
        return r


chromosomes = '2R', '2L', '3R', '3L', 'X', 'Y_unplaced', 'UNKN'
autosomes = chromosomes[:4]


gene_labels = {
    'AGAP009195': 'Gste1',
    'AGAP009194': 'Gste2',
    'AGAP009197': 'Gste3',
    'AGAP009193': 'Gste4',
    'AGAP009192': 'Gste5',
    'AGAP009191': 'Gste6',
    'AGAP009196': 'Gste7',
    'AGAP009190': 'Gste8',
    'AGAP004707': 'Vgsc',
    'AGAP002862': 'Cyp6aa1',
    'AGAP013128': 'Cyp6aa2',
    'AGAP002863': 'Coeae6o',
    'AGAP002865': 'Cyp6p3',
    'AGAP002866': 'Cyp6p5',
    'AGAP002867': 'Cyp6p4',
    'AGAP002868': 'Cyp6p1',
    'AGAP002869': 'Cyp6p2',
    'AGAP002870': 'Cyp6ad1',
    'AGAP002915': 'Pcsk4/furin',
    'AGAP002825': 'Pp01',
    'AGAP002824': 'Gprtak1',
    'AGAP006028': 'Gaba',
    'AGAP010815': 'Tep1'
}


def get_geneset_features(geneset_fn, chrom, start=None, stop=None):
    """Function to load geneset features for a specific genome region via petl."""
    if start and stop:
        region = '%s:%s-%s' % (chrom, start, stop)
    else:
        region = chrom
    return etl.fromgff3(geneset_fn, region=region)


def plot_genes(genome, geneset_fn, chrom, start=1, stop=None, ax=None, height=.3, label=False,
               labels=None, label_unnamed=True, label_rotation=45, barh_kwargs=None):

    if stop is None:
        stop = len(genome[chrom])

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 1))
        sns.despine(ax=ax, offset=5)

    genes = get_geneset_features(geneset_fn, chrom, start, stop).eq('type', 'gene').records()

    fwd_ranges = [(g.start, (g.end - g.start)) for g in genes if g.strand == '+']
    rev_ranges = [(g.start, (g.end - g.start)) for g in genes if g.strand == '-']
    if barh_kwargs is None:
        barh_kwargs = dict()
    barh_kwargs.setdefault('color', 'k')
    ax.broken_barh(fwd_ranges, (.5, height), **barh_kwargs)
    ax.broken_barh(rev_ranges, (.5-height, height), **barh_kwargs)
    ax.set_ylim(0, 1)
    ax.axhline(.5, color='k', linestyle='-')
    ax.set_xlim(start, stop)
    ax.set_yticks([.5-(height/2), .5+(height/2)])
    ax.set_yticklabels(['-', '+'])
    ax.set_ylabel('genes', rotation=0, ha='right', va='center')

    if label:
        for gene in genes:
            gid = gene.attributes['ID']
            if labels and gid not in labels and not label_unnamed:
                continue
            if labels and gid in labels:
                label = labels[gid]
            else:
                label = gid
            x = gene.start
            if x < start:
                x = start
            if x > stop:
                x = stop
            if gene.strand == '+':
                rotation = label_rotation
                y = .5 + height
                ax.text(x, y, label, rotation=rotation, fontsize=6, ha='left', va='bottom')
            else:
                rotation = -label_rotation
                y = .5 - height
                ax.text(x, y, label, rotation=rotation, fontsize=6, ha='left', va='top')
