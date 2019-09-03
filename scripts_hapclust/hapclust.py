# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division
import bisect
import collections


import numpy as np
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
import allel
import graphviz
import pyximport
pyximport.install(setup_args=dict(include_dirs=np.get_include()),
                  reload_support=True)
from hapclust_opt import count_gametes


#########################
# HIERARCHICAL CLUSTERING
#########################


def get_descendant(node, desc_id):
    """Search the descendants of the given node in a scipy tree.

    Parameters
    ----------
    node : scipy.cluster.hierarchy.ClusterNode
        The ancestor node to search from.
    desc_id : int
        The ID of the node to search for.

    Returns
    -------
    desc : scipy.cluster.hierarchy.ClusterNode
        If a node with the given ID is not found, returns None.

    """
    if node.id == desc_id:
        return node
    if node.is_leaf():
        return None
    if node.left.id == desc_id:
        return node.left
    if node.right.id == desc_id:
        return node.right
    # search left
    l = get_descendant(node.left, desc_id)
    if l is not None:
        return l
    # search right
    r = get_descendant(node.right, desc_id)
    return r


# monkey-patch as a method
scipy.cluster.hierarchy.ClusterNode.get_descendant = get_descendant


def fig_haplotypes_clustered(h,
                             distance_metric='hamming',
                             linkage_method='single',
                             truncate_distance=0,
                             orientation='top',
                             subplot_ratios=(4, 2),
                             subplot_pad=0,
                             despine_offset=5,
                             count_sort=True,
                             dend_linecolor='k',
                             cut_height=2,
                             highlight_clusters=True,
                             highlight_colors=None,
                             highlight_dend=True,
                             highlight_freq=True,
                             highlight_alpha=0.3,
                             label_clusters=True,
                             dpi=None,
                             fig=None,
                             ):
    """Construct a plot of hierarchical clustering of haplotypes.

    Parameters
    ----------
    TODO

    Returns
    -------
    TODO

    """

    # check inputs
    h = allel.HaplotypeArray(h)

    # compute distance matrix
    dist = scipy.spatial.distance.pdist(h.T, metric=distance_metric)
    if distance_metric in {'hamming', 'jaccard'}:
        # convert distance to number of SNPs, easier to interpret
        dist *= h.n_variants

    # compute hierarchical clustering
    Z = scipy.cluster.hierarchy.linkage(dist, method=linkage_method)

    # Z is a linkage matrix. From the scipy docs...
    # A 4 by (n-1) matrix Z is returned. At the i-th iteration, clusters with
    # indices Z[i, 0] and Z[i, 1] are combined to form cluster n + i. A cluster
    # with an index less than n corresponds to one of the original observations.
    # The distance between clusters Z[i, 0] and Z[i, 1] is given by Z[i, 2]. The
    # fourth value Z[i, 3] represents the number of original observations in the
    # newly formed cluster.

    # find level to truncate dendrogram
    lastp = h.n_haplotypes - bisect.bisect_right(Z[:, 2], truncate_distance)

    # convenience variables
    horizontal = orientation in ['left', 'right']
    vertical = not horizontal
    inverted = orientation in ['bottom', 'right']

    # setup figure
    if fig is None:
        figsize = plt.rcParams['figure.figsize']
        if horizontal:
            figsize = figsize[::-1]
        fig = plt.figure(figsize=figsize, dpi=dpi)

    # setup gridspec and axes
    if inverted:
        subplot_ratios = subplot_ratios[::-1]
    if horizontal:
        gs = plt.GridSpec(nrows=1, ncols=2, width_ratios=subplot_ratios)
    else:
        gs = plt.GridSpec(nrows=2, ncols=1, height_ratios=subplot_ratios)
    if inverted:
        ax_dend = fig.add_subplot(gs[1])
        ax_freq = fig.add_subplot(gs[0])
    else:
        ax_dend = fig.add_subplot(gs[0])
        ax_freq = fig.add_subplot(gs[1])
    if horizontal:
        sns.despine(ax=ax_dend, offset=despine_offset,
                    left=True, top=True, right=True, bottom=False)
        sns.despine(ax=ax_freq, offset=despine_offset,
                    left=True, top=True, right=True, bottom=False)
    else:
        sns.despine(ax=ax_dend, offset=despine_offset,
                    left=False, top=True, right=True, bottom=True)
        sns.despine(ax=ax_freq, offset=despine_offset,
                    left=False, top=True, right=True, bottom=True)

    # make a dendrogram
    kwargs_dend = dict(
        truncate_mode='lastp',
        p=lastp,
        show_leaf_counts=False,
        count_sort=count_sort,
        no_labels=True,
        color_threshold=0,
        above_threshold_color=dend_linecolor,
        orientation=orientation
    )
    dend = scipy.cluster.hierarchy.dendrogram(Z, ax=ax_dend, **kwargs_dend)
    leaves = dend['leaves']
    ax_dend_label = 'Distance'
    if horizontal:
        ax_dend.set_xlabel(ax_dend_label)
        ax_dend.set_yticks([])
    else:
        ax_dend.set_ylabel(ax_dend_label)
        ax_dend.set_xticks([])

    # construct a tree and compute observation counts for the dendrogram leaves
    tree = scipy.cluster.hierarchy.to_tree(Z)
    s = np.arange(len(leaves))
    t = np.array([
        1 if l < h.n_haplotypes
        else tree.get_descendant(l).get_count()
        for l in leaves
    ])

    # plot frequencies bar
    ax_freq_label = 'Frequency'
    if horizontal:
        ax_freq.barh(s, t, height=1, lw=0, color='k', align='edge')
        ax_freq.set_ylim(0, len(leaves))
        ax_freq.set_yticks([])
        ax_freq.set_xlabel(ax_freq_label)
        ax_freq.grid(axis='x', lw=.5)
        if orientation == 'right':
            ax_freq.invert_xaxis()
        # remove 0
        ax_freq.set_xticks(ax_freq.get_xticks()[1:])
    else:
        ax_freq.bar(s, t, width=1, lw=0, color='k', align='edge')
        ax_freq.set_xlim(0, len(leaves))
        ax_freq.set_xticks([])
        ax_freq.set_ylabel(ax_freq_label)
        ax_freq.grid(axis='y', lw=.5)
        if orientation == 'top':
            ax_freq.invert_yaxis()
        # remove 0
        ax_freq.set_yticks(ax_freq.get_yticks()[1:])

    # cut the tree
    cut = scipy.cluster.hierarchy.cut_tree(Z, height=cut_height)[:, 0]
    cluster_sizes = np.bincount(cut)
    clusters = [np.nonzero(cut == i)[0] for i in range(cut.max() + 1)]

    # now the fiddly bit - we need to figure out where the clusters have
    # ended up in the dendrogram we plotted earlier...

    # N.B., the dendrogram was truncated, so each leaf in the dendrogram
    # may correspond to more than one original observation (i.e., haplotype).
    # Let's build a list storing the observations for each leaf:
    leaf_obs = [tree.get_descendant(ix).pre_order() for ix in leaves]

    # Now let's figure out for each leaf in the dendrogram, which of the clusters
    # obtained by cutting the tree earlier does it fall into?
    leaf_clusters = np.array([cut[l[0]] for l in leaf_obs])

    # Now let's build a data structure that reorders the clusters so they
    # occur in the same order as in the dendrogram, and also record the indices
    # of the start and stop leaf for each cluster:
    cluster_spans = list()
    c_prv = leaf_clusters[0]
    i_start = 0
    for i, c in enumerate(leaf_clusters[1:], 1):
        if c != c_prv:
            cluster_spans.append((i_start, i, clusters[c_prv]))
            i_start = i
        c_prv = c
    # don't forget the last one
    cluster_spans.append((i_start, i+1, clusters[c]))

    # OK, now figure out which clusters we want to highlight...
    if isinstance(highlight_clusters, (list, tuple)):
        # user has manually specified which clusters to highlight
        pass
    else:
        # assume highlight_clusters is the minimum cluster size to highlight
        min_cluster_size = int(highlight_clusters)
        highlight_clusters = [i for i, cs in enumerate(cluster_spans)
                              if len(cs[2]) >= min_cluster_size]

    # setup colors for highlighting clusters
    if highlight_colors is None:
        highlight_colors = sns.color_palette('hls', n_colors=len(highlight_clusters))

    # do the highlighting
    for color, cix in zip(highlight_colors, highlight_clusters):
        start, stop, _ = cluster_spans[cix]
        if horizontal:
            freq_spanf = ax_freq.axhspan
            dend_patch_xy = (0, start * 10)
            dend_patch_width = cut_height
            dend_patch_height = (stop - start) * 10
        else:
            freq_spanf = ax_freq.axvspan
            dend_patch_xy = (start * 10, 0)
            dend_patch_width = (stop - start) * 10
            dend_patch_height = cut_height
        if highlight_freq:
            freq_spanf(start, stop, color=color, alpha=highlight_alpha, zorder=-20)
        if highlight_dend:
            ax_dend.add_patch(plt.Rectangle(xy=dend_patch_xy,
                                            width=dend_patch_width,
                                            height=dend_patch_height,
                                            color=color, alpha=highlight_alpha,
                                            zorder=-20))

    # for debugging, label the clusters
    if label_clusters:
        for i, (start, stop, clst) in enumerate(cluster_spans):
            if horizontal:
                x = max(ax_freq.get_xlim())
                y = (start + stop) / 2
                ha = orientation
                va = 'center'
            else:
                x = (start + stop) / 2
                y = max(ax_freq.get_ylim())
                ha = 'center'
                va = orientation
            # treat label_clusters as minimum cluster size to label
            if len(clst) >= int(label_clusters):
                ax_freq.text(x, y, str(i),
                             va=va, ha=ha, fontsize=6)

    # tidy up plot
    if horizontal:
        gs.tight_layout(fig, w_pad=subplot_pad)
    else:
        gs.tight_layout(fig, h_pad=subplot_pad)

    # return some useful stuff
    return fig, ax_dend, ax_freq, cluster_spans, leaf_obs


####################
# HAPLOTYPE NETWORKS
####################


def _graph_edges(graph,
                 edges,
                 hap_counts,
                 node_size_factor,
                 edge_length,
                 anon_width,
                 intermediate_nodes,
                 edge_attrs,
                 anon_node_attrs,
                 h_distinct,
                 variant_labels):

    for i in range(edges.shape[0]):

        for j in range(edges.shape[1]):

            # lookup distance between nodes i and j
            sep = edges[i, j]

            if sep > 0:

                # lookup number of haplotypes
                # calculate node sizes (needed to adjust edge length)
                if i < len(hap_counts):
                    # original observation
                    n_i = hap_counts[i]
                    width_i = np.sqrt(n_i * node_size_factor)
                else:
                    # not an original observation
                    n_i = 1
                    width_i = anon_width
                if j < len(hap_counts):
                    # original observation
                    n_j = hap_counts[j]
                    width_j = np.sqrt(n_j * node_size_factor)
                else:
                    # not an original observation
                    n_j = 1
                    width_j = anon_width

                if sep > 1 and intermediate_nodes:

                    # tricky case, need to add some anonymous nodes to represent
                    # intermediate steps

                    # handle variant labels
                    if variant_labels is not None:
                        idx_diff = np.nonzero(h_distinct[:, i] != h_distinct[:, j])[0]
                        labels = variant_labels[idx_diff]
                        reverse = h_distinct[idx_diff, i] > h_distinct[idx_diff, j]
                    else:
                        labels = [''] * sep

                    # add first intermediate node
                    nid = 'anon_{}_{}_{}'.format(i, j, 0)
                    graph.node(nid, label='', width=str(anon_width), **anon_node_attrs)

                    # add edge from node i to first intermediate
                    el = edge_length + width_i / 2 + anon_width / 2
                    edge_from, edge_to = str(i), 'anon_{}_{}_{}'.format(i, j, 0)
                    kwargs = {'len': str(el), 'label': labels[0]}
                    kwargs.update(edge_attrs)
                    if labels[0]:
                        # this will be a directed edge
                        del kwargs['dir']
                        kwargs.setdefault('arrowsize', '0.5')
                        if reverse[0]:
                            edge_from, edge_to = edge_to, edge_from
                    graph.edge(edge_from, edge_to, **kwargs)

                    # add further intermediate nodes as necessary
                    for k in range(1, sep-1):
                        edge_from, edge_to = ('anon_{}_{}_{}'.format(i, j, k-1),
                                              'anon_{}_{}_{}'.format(i, j, k))
                        graph.node(edge_to, label='', width=str(anon_width),
                                   **anon_node_attrs)
                        el = edge_length + anon_width
                        kwargs = {'len': str(el), 'label': labels[k]}
                        kwargs.update(edge_attrs)
                        if labels[k]:
                            # this will be a directed edge
                            del kwargs['dir']
                            kwargs.setdefault('arrowsize', '0.5')
                            if reverse[k]:
                                edge_from, edge_to = edge_to, edge_from
                        graph.edge(edge_from, edge_to, **kwargs)

                    # add edge from final intermediate node to node j
                    edge_from, edge_to = 'anon_{}_{}_{}'.format(i, j, sep-2), str(j)
                    el = edge_length + anon_width / 2 + width_j / 2
                    kwargs = {'len': str(el), 'label': labels[-1]}
                    kwargs.update(edge_attrs)
                    if labels[-1]:
                        # this will be a directed edge
                        del kwargs['dir']
                        kwargs.setdefault('arrowsize', '0.5')
                        if reverse[-1]:
                            edge_from, edge_to = edge_to, edge_from
                    graph.edge(edge_from, edge_to, **kwargs)

                else:

                    # simple case, direct edge from node i to j

                    # N.B., adjust edge length so we measure distance from edge of
                    # circle rather than center
                    el = (edge_length * sep) + width_i / 2 + width_j / 2
                    kwargs = {'len': str(el)}
                    kwargs.update(edge_attrs)
                    edge_from, edge_to = str(i), str(j)

                    if variant_labels is not None:
                        idx_diff = np.nonzero(h_distinct[:, i] != h_distinct[:, j])[0][0]
                        label = variant_labels[idx_diff]
                        if label:
                            # this will be a directed edge
                            del kwargs['dir']
                            kwargs.setdefault('arrowsize', '0.5')
                            allele_i = h_distinct[idx_diff, i]
                            allele_j = h_distinct[idx_diff, j]
                            if allele_i > allele_j:
                                # reverse direction of edge
                                edge_from, edge_to = edge_to, edge_from
                    else:
                        label = ''
                    kwargs.setdefault('label', label)
                    graph.edge(edge_from, edge_to, **kwargs)


import sys


class DummyLogger(object):

    def __call__(self, *args, **kwargs):
        pass


class DebugLogger(object):

    def __init__(self, name, out=None):
        self.name = name
        if out is None:
            out = sys.stdout
        elif isinstance(out, str):
            out = open(out, mode='at')
        self.out = out

    def __call__(self, *msg):
        print(self.name, *msg, file=self.out)
        self.out.flush()


def _pairwise_haplotype_distance(h, metric='hamming'):
    assert metric in ['hamming', 'jaccard']
    dist = allel.pairwise_distance(h, metric=metric)
    dist *= h.n_variants
    dist = scipy.spatial.distance.squareform(dist)
    # N.B., np.rint is **essential** here, otherwise can get weird rounding errors
    dist = np.rint(dist).astype('i8')
    return dist


def graph_haplotype_network(h,
                            hap_colors='grey',
                            distance_metric='hamming',
                            network_method='mjn',
                            comment=None,
                            engine='neato',
                            format='png',
                            mode='major',
                            overlap=True,
                            splines=True,
                            graph_attrs=None,
                            node_size_factor=0.005,
                            node_attrs=None,
                            show_node_labels=False,
                            fontname='monospace',
                            fontsize=None,
                            edge_length=0.5,
                            edge_weight=10,
                            edge_attrs=None,
                            show_alternate_edges=True,
                            alternate_edge_attrs=None,
                            anon_width=0.03,
                            anon_fillcolor='white',
                            anon_node_attrs=None,
                            intermediate_nodes=True,
                            max_dist=5,
                            variant_labels=None,
                            debug=False,
                            debug_out=None,
                            max_allele=3,
                            return_components=False,
                            show_singletons=True,
                            ):
    """TODO doc me"""

    if debug:
        log = DebugLogger('[graph_haplotype_network]', out=debug_out)
    else:
        log = DummyLogger()

    # check inputs
    h = allel.HaplotypeArray(h)
    log(h.shape)

    # optimise - keep only segregating variants
    ac = h.count_alleles()
    loc_seg = ac.is_segregating()
    h = h[loc_seg]
    if variant_labels is not None:
        variant_labels = np.asarray(variant_labels, dtype=object)[loc_seg]

    # find distinct haplotypes
    h_distinct_sets = h.distinct()
    # log('h_distinct_sets', h_distinct_sets)

    # find indices of distinct haplotypes - just need one per set
    h_distinct_indices = [sorted(s)[0] for s in h_distinct_sets]
    log('h_distinct_indices', h_distinct_indices)

    # reorder by index
    ix = np.argsort(h_distinct_indices)
    h_distinct_indices = [h_distinct_indices[i] for i in ix]
    log('h_distinct_indices (reordered)', h_distinct_indices)
    h_distinct_sets = [h_distinct_sets[i] for i in ix]

    # obtain an array of distinct haplotypes
    h_distinct = h.take(h_distinct_indices, axis=1)

    # deal with colors - count how many of each color per distinct haplotype
    color_counters = None
    if isinstance(hap_colors, (list, tuple, np.ndarray)):
        assert len(hap_colors) == h.n_haplotypes
        color_counters = [
            collections.Counter([hap_colors[i] for i in s])
            for s in h_distinct_sets
        ]

    # count how many observations per distinct haplotype
    hap_counts = [len(s) for s in h_distinct_sets]

    # compute pairwise distance matrix
    dist = _pairwise_haplotype_distance(h_distinct, distance_metric)

    if network_method.lower() == 'mst':

        # compute minimum spanning tree
        edges = scipy.sparse.csgraph.minimum_spanning_tree(dist).toarray().astype(int)

        # deal with maximum distance
        if max_dist:
            edges[edges > max_dist] = 0

        # no alternate edges when using mst
        alternate_edges = None

    elif network_method.lower() == 'msn':

        # compute network
        edges, alternate_edges = minimum_spanning_network(dist,
                                                          max_dist=max_dist,
                                                          debug=debug,
                                                          debug_out=debug_out)
        edges = np.triu(edges)
        alternate_edges = np.triu(alternate_edges)

    elif network_method.lower() == 'mjn':

        # compute network - N.B., MJN may add new haplotypes
        h_distinct, edges, alternate_edges = median_joining_network(h_distinct,
                                                                    max_dist=max_dist,
                                                                    debug=debug,
                                                                    debug_out=debug_out,
                                                                    max_allele=max_allele)
        edges = np.triu(edges)
        alternate_edges = np.triu(alternate_edges)

    else:
        raise ValueError(network_method)

    # setup graph
    graph = graphviz.Digraph(comment=comment, engine=engine, format=format)
    if graph_attrs is None:
        graph_attrs = dict()
    graph_attrs.setdefault('overlap', str(overlap).lower())
    graph_attrs.setdefault('splines', str(splines).lower())
    graph_attrs.setdefault('mode', mode)
    graph_attrs.setdefault('sep', '0')
    graph.attr('graph', **graph_attrs)

    # add the main nodes
    if node_attrs is None:
        node_attrs = dict()
    node_attrs.setdefault('fixedsize', 'true')
    node_attrs.setdefault('shape', 'circle')
    node_attrs.setdefault('fontname', fontname)
    node_attrs.setdefault('fontsize', str(fontsize))
    if anon_node_attrs is None:
        anon_node_attrs = dict()
    anon_node_attrs.setdefault('fixedsize', 'true')
    anon_node_attrs.setdefault('shape', 'circle')
    anon_node_attrs.setdefault('style', 'filled')
    anon_node_attrs.setdefault('fillcolor', anon_fillcolor)
    anon_node_attrs.setdefault('fontname', fontname)
    anon_node_attrs.setdefault('fontsize', str(fontsize))
    for i in range(edges.shape[0]):
        kwargs = dict()

        if i < len(hap_counts):
            # original haplotype

            n = hap_counts[i]
            connected = np.any((edges[i] > 0) | (edges[:, i] > 0))
            if not show_singletons and n == 1 and not connected:
                continue

            # calculate width from number of items - make width proportional to area
            width = np.sqrt(n * node_size_factor)

            # determine style and fill color
            if color_counters:
                cc = color_counters[i]
                if len(cc) > 1:
                    # more than one color, make a pie chart
                    style = 'wedged'
                    fillcolor = ':'.join(['%s;%s' % (k, v/n) for k, v in cc.items()])
                else:
                    # just one color, fill with solid color
                    style = 'filled'
                    fillcolor = list(cc.keys())[0]
            else:
                style = 'filled'
                fillcolor = hap_colors

            kwargs.update(node_attrs)
            kwargs.setdefault('style', style)
            kwargs.setdefault('fillcolor', fillcolor)
            kwargs.setdefault('width', str(width))

        else:
            # not an original haplotype, inferred during network building

            n = 1

            width = anon_width
            fillcolor = anon_fillcolor
            kwargs.update(anon_node_attrs)
            kwargs.setdefault('width', str(anon_width))

        # add graph node
        if show_node_labels is False:
            label = ''
        elif show_node_labels is True:
            label = str(i)
        elif isinstance(show_node_labels, int) and n >= show_node_labels:
            label = str(i)
        elif show_node_labels == 'count' and n > 1:
            label = str(n)
        else:
            label = ''
        kwargs.setdefault('label', label)
        graph.node(str(i), **kwargs)

    # setup defaults
    if edge_attrs is None:
        edge_attrs = dict()
    edge_attrs.setdefault('style', 'normal')
    edge_attrs.setdefault('weight', str(edge_weight))
    edge_attrs.setdefault('fontname', fontname)
    edge_attrs.setdefault('fontsize', str(fontsize))
    edge_attrs.setdefault('dir', 'none')
    if alternate_edge_attrs is None:
        alternate_edge_attrs = dict()
    alternate_edge_attrs.setdefault('style', 'dashed')
    alternate_edge_attrs.setdefault('weight', str(edge_weight))
    alternate_edge_attrs.setdefault('fontname', fontname)
    alternate_edge_attrs.setdefault('fontsize', str(fontsize))
    alternate_edge_attrs.setdefault('dir', 'none')

    # add main edges
    _graph_edges(graph,
                 edges,
                 hap_counts,
                 node_size_factor,
                 edge_length,
                 anon_width,
                 intermediate_nodes,
                 edge_attrs,
                 anon_node_attrs,
                 h_distinct,
                 variant_labels)

    # add alternate edges
    if show_alternate_edges and alternate_edges is not None:
        _graph_edges(graph,
                     alternate_edges,
                     hap_counts,
                     node_size_factor,
                     edge_length,
                     anon_width,
                     intermediate_nodes,
                     alternate_edge_attrs,
                     anon_node_attrs,
                     h_distinct,
                     variant_labels)

    if return_components:
        from scipy.sparse.csgraph import connected_components
        n_components, component_labels = connected_components(edges)
        return graph, h_distinct_sets, component_labels

    else:
        return graph, hap_counts


def minimum_spanning_network(dist, max_dist=None, debug=False, debug_out=None):
    """TODO"""

    if debug:
        log = DebugLogger('[minimum_spanning_network]', out=debug_out)
    else:
        log = DummyLogger()

    # TODO review implementation, see if this can be tidied up

    # keep only the upper triangle of the distance matrix, to avoid adding the same
    # edge twice
    dist = np.triu(dist)

    # setup the output array of links between nodes
    edges = np.zeros_like(dist)

    # setup an array of alternate links
    alternate_edges = np.zeros_like(dist)

    # intermediate variable - assignment of haplotypes to clusters (a.k.a. sub-networks)
    # initially each distinct haplotype is in its own cluster
    cluster = np.arange(dist.shape[0])

    # start with haplotypes separated by a single mutation
    step = 1
    log('[%s]' % step, 'begin')

    # iterate until all haplotypes in a single cluster, or max_dist reached
    while len(set(cluster)) > 1 and (max_dist is None or step <= max_dist):
        log('[%s]' % step, 'processing, cluster:', cluster)

        # keep track of which clusters have been merged at this height
        merged = set()

        # remember what cluster assignments were at the previous height
        prv_cluster = cluster.copy()

        # iterate over all pairs where distance equals current step size
        for i, j in zip(*np.nonzero(dist == step)):
            log('[%s]' % step, 'found potential edge', i, j)

            # current cluster assignment for each haplotype
            a = cluster[i]
            b = cluster[j]

            # previous cluster assignment for each haplotype
            pa = prv_cluster[i]
            pb = prv_cluster[j]

            log('[%s]' % step, a, b, pa, pb, merged)

            # check to see if both nodes already in the same cluster
            if a != b:

                # nodes are in different clusters, so we can merge (i.e., connect) the
                # clusters

                log('[%s]' % step, 'assign an edge')
                edges[i, j] = dist[i, j]
                edges[j, i] = dist[i, j]

                # merge clusters
                c = cluster.max() + 1
                loc_a = cluster == a
                loc_b = cluster == b
                cluster[loc_a] = c
                cluster[loc_b] = c
                merged.add(tuple(sorted([pa, pb])))
                log('[%s]' % step, 'merged', cluster, merged)

            elif tuple(sorted([pa, pb])) in merged or step == 1:

                # the two clusters have already been merged at this level, this is an
                # alternate connection
                # N.B., special case step = 1 because no previous cluster assignments
                # (TODO really?)

                log('[%s]' % step, 'assign an alternate edge')
                alternate_edges[i, j] = dist[i, j]
                alternate_edges[j, i] = dist[i, j]

            else:

                log('[%s]' % step, 'WTF?')

        # increment step
        step += 1

    log('# edges:', np.count_nonzero(np.triu(edges)))
    log('# alt edges:', np.count_nonzero(np.triu(alternate_edges)))
    return edges, alternate_edges


def _remove_obsolete(h, orig_n_haplotypes, max_dist, log):
    n_removed = None
    edges = alt_edges = None

    while n_removed is None or n_removed > 0:

        # step 1 - compute distance
        dist = _pairwise_haplotype_distance(h, metric='hamming')

        # step 2 - construct the minimum spanning network
        edges, alt_edges = minimum_spanning_network(dist, max_dist=max_dist)
        all_edges = edges + alt_edges

        # step 3 - remove obsolete sequence types
        loc_keep = np.ones(h.n_haplotypes, dtype=bool)
        for i in range(orig_n_haplotypes, h.n_haplotypes):
            n_connections = np.count_nonzero(all_edges[i])
            if n_connections <= 2:
                loc_keep[i] = False
        n_removed = np.count_nonzero(~loc_keep)
        log('discarding', n_removed, 'obsolete haplotypes')
        h = h[:, loc_keep]

    return h, edges, alt_edges


def median_joining_network(h, max_dist=None, debug=False, debug_out=None, max_allele=3):
    """TODO doc me"""

    if debug:
        log = DebugLogger('[median_joining_network]', out=debug_out)
    else:
        log = DummyLogger()

    h = allel.HaplotypeArray(h, dtype='i1')
    orig_n_haplotypes = h.n_haplotypes

    n_medians_added = None
    iteration = 0
    while n_medians_added is None or n_medians_added > 0:

        # steps 1-3
        h, edges, alt_edges = _remove_obsolete(h, orig_n_haplotypes, max_dist=max_dist,
                                               log=log)
        all_edges = edges + alt_edges

        # step 4 - add median vectors

        # iterate over all triplets
        n = h.n_haplotypes
        seen = set([hash(h[:, i].tobytes()) for i in range(h.n_haplotypes)])
        new_haps = list()
        for i in range(n):
            for j in range(i + 1, n):
                if all_edges[i, j]:
                    for k in range(n):
                        if all_edges[i, k] or all_edges[j, k]:
                            log(iteration, i, j, k, 'computing median vector')
                            uvw = h[:, [i, j, k]]
                            ac = uvw.count_alleles(max_allele=max_allele)
                            # majority consensus haplotype
                            x = np.argmax(ac, axis=1).astype('i1')
                            x_hash = hash(x.tobytes())
                            log(iteration, i, j, k, 'median vector', x)
                            # test if x already in haps
                            if x_hash in seen:
                                log(iteration, i, j, k, 'median vector already present')
                                pass
                            else:
                                log(iteration, i, j, k, 'adding median vector')
                                new_haps.append(x.tolist())
                                seen.add(x_hash)
        n_medians_added = len(new_haps)
        log(new_haps)
        if n_medians_added:
            h = h.concatenate(allel.HaplotypeArray(np.array(new_haps, dtype='i1').T),
                              axis=1)

        iteration += 1

    # final pass
    h, edges, alt_edges = _remove_obsolete(h, orig_n_haplotypes, max_dist=max_dist,
                                           log=log)
    return h, edges, alt_edges


def locate_recombinants(h, debug=False):
    """Locate recombinant haplotypes via the four gamete test."""
    count = count_gametes(np.asarray(h, dtype='i1'))
    d = np.all(count > 0, axis=(2, 3))
    # indices of recombinant haplotypes - N.B., keep track of different possible solutions
    # because we want to find the smallest number of haplotypes to remove
    solutions = [set()]
    # find indices of pairs of variants with evidence for recombination
    for i, j in zip(*np.nonzero(d)):
        # find the least frequent gametic type
        min_count = np.min(count[i, j])
        new_solutions = []
        for least_frequent_gamete in zip(*np.nonzero(count[i, j] == min_count)):
            # find indices of haplotypes of the least frequent gametic type
            recombinant_haps_idx = set(
                np.nonzero(
                    np.all(h[[i, j], :] == np.array(least_frequent_gamete)[:, np.newaxis], axis=0)
                )[0]
            )
            if debug:
                print(i, j, count[i, j].flatten(), least_frequent_gamete, recombinant_haps_idx)
            new_solutions.extend([s.union(recombinant_haps_idx) for s in solutions])
        solutions = new_solutions
    # sort solutions by size
    return sorted(solutions, key=lambda s: len(s))


# MISC PLOTTING
###############


import matplotlib.image as mpimg
import io


def plot_graphviz(graph, ax, size=None, desired_size=False, ratio=None, dpi=None,
                  interpolation='bilinear', aspect='equal'):
    """Plot a graphviz graph onto a matplotlib axes object."""
    fig = ax.figure
    if size is None:
        # try to match size to ax
        fw, fh = fig.get_figwidth(), fig.get_figheight()
        bbox = ax.get_position()
        w = fw * bbox.width
        h = fh * bbox.height
        size = w, h
    if dpi is None:
        # match dpi to fig
        dpi = fig.dpi

    # set size and resolution
    size = '%s,%s' % (w, h)
    if desired_size:
        size += '!'
    kwargs = dict(
        size=size,
        dpi=str(dpi),
    )
    if ratio:
        kwargs['ratio'] = str(ratio)
    else:
        kwargs['ratio'] = ''
    graph.attr('graph', **kwargs)

    # render the graph as png
    dat = graph.pipe(format='png')

    # read the png data into an image array
    img = mpimg.imread(io.BytesIO(dat))

    # plot the image
    ax.imshow(img, interpolation=interpolation, aspect=aspect)


import matplotlib as mpl


def plot_haplotypes(ax, h, variant_labels=None, colors=('w', 'k'), vmin=0, vmax=2):
    cmap = mpl.colors.ListedColormap(colors, name='mymap')
    ax.pcolormesh(np.asarray(h[::-1]), cmap=cmap, vmin=vmin, vmax=vmax)
    if variant_labels:
        ax.set_yticks(np.arange(h.shape[0])+.5)
        ax.set_yticklabels(variant_labels[::-1], family='monospace')
        ax.hlines(np.arange(h.shape[0]+1), 0, h.shape[1], color='k', lw=.5)
    ax.set_xlim(0, h.shape[1])
    ax.set_ylim(0, h.shape[0])


# HAPLOTYPE SHARING
###################


def split_flanks(h, pos, pos_core):
    """Split a haplotype array into two flanks on some core position."""

    h = np.asarray(h)
    pos = np.asarray(pos)
    idx_split = bisect.bisect_left(pos, pos_core)
    dist_right = pos[idx_split:] - pos_core
    haps_right = allel.HaplotypeArray(h[idx_split:, :])
    # reverse so both flanks have same orientation w.r.t core
    dist_left = pos_core - pos[:idx_split][::-1]
    haps_left = allel.HaplotypeArray(h[:idx_split, :][::-1, :])
    return dist_right, dist_left, haps_right, haps_left


def neighbour_haplotype_sharing(haps_ehh, haps_mut, dist_ehh, dist_mut, jitter=False):
    """Analyse sharing between haplotypes in prefix sorted order."""

    haps_ehh = allel.HaplotypeArray(haps_ehh)
    haps_mut = allel.HaplotypeArray(haps_mut)
    n_haplotypes = haps_ehh.n_haplotypes
    assert n_haplotypes == haps_mut.n_haplotypes

    # sort by prefix
    idx_sorted = haps_ehh.prefix_argsort()
    haps_ehh_sorted = haps_ehh[:, idx_sorted]
    haps_mut_sorted = haps_mut[:, idx_sorted]

    # compute length (no. variants) of shared prefix between neighbours
    nspl = allel.opt.stats.neighbour_shared_prefix_lengths(
        np.asarray(haps_ehh_sorted, dtype='i1')
    )

    # compute length (physical distance)
    nspd = _shared_distance(nspl, dist_ehh, jitter=jitter)

    # compute number of mutations on shared haplotypes
    muts = np.zeros_like(nspl)
    for i in range(n_haplotypes - 1):
        # distance at which haplotypes diverge
        d = nspd[i]
        # index into mutations array where haplotypes diverge
        ix = bisect.bisect_right(dist_mut, d)
        # number of mutations
        m = np.count_nonzero(
            haps_mut_sorted[:ix, i] != haps_mut_sorted[:ix, i+1])
        muts[i] = m

    return idx_sorted, nspl, nspd, muts


def haplotype_accessible_length(spd, core_pos, is_accessible, flank):
    spd_accessible = np.zeros_like(spd)
    for i in range(spd.shape[0]):
        d = spd[i]
        if flank == 'right':
            da = np.count_nonzero(is_accessible[core_pos - 1:core_pos + d - 1])
        else:
            da = np.count_nonzero(is_accessible[core_pos - d - 1:core_pos - 1])
        spd_accessible[i] = da
    return spd_accessible


def fig_neighbour_haplotype_sharing(nspd, muts, haps_display, pop_colors,
                                    nspd_accessible=None,
                                    haps_display_vlbl=None,
                                    nspd_cut=None,
                                    nspd_ylim=None,
                                    nspd_yscale='log',
                                    cluster_palette=None,
                                    muts_ylim=None,
                                    that_ylim=None,
                                    mu=3.5e-9,
                                    rr=1e-8,
                                    gs_height_ratios=(6, 3, 3, .5, 6),
                                    fig=None):

    # check args
    nspd = np.asarray(nspd)
    muts = np.asarray(muts)
    haps_display = allel.HaplotypeArray(haps_display)
    n_haplotypes = nspd.shape[0] + 1
    assert n_haplotypes == muts.shape[0] + 1
    assert n_haplotypes == haps_display.n_haplotypes

    # setup figure
    if fig is None:
        fig = plt.figure()
    gs = mpl.gridspec.GridSpec(nrows=5, ncols=1, height_ratios=gs_height_ratios)
    if cluster_palette is None:
        cluster_palette = sns.color_palette('Set3', n_colors=12)

    # plot shared haplotype lengths
    ###############################

    ax_nspd = fig.add_subplot(gs[0])
    sns.despine(ax=ax_nspd, offset=5, bottom=True)
    # N.B., values are distances shared between neighbouring haplotypes
    x = np.arange(0, n_haplotypes - 1) + .5
    y = nspd
    ax_nspd.plot(x, y, color='k', lw=.5, zorder=10)

    # clustering
    if nspd_cut:
        clst_start_idx = 0
        clst_idx = 0
        for i in range(1, n_haplotypes):
            if y[i-1] < nspd_cut:
                if i - clst_start_idx > 1:
                    color = cluster_palette[clst_idx % len(cluster_palette)]
                    ax_nspd.fill_between(x[clst_start_idx:i], 0, y[clst_start_idx:i], color=color)
                    clst_idx += 1
                clst_start_idx = i
        ax_nspd.axhline(nspd_cut, color='k', linestyle='--', lw=.5)

    # tidy up
    if nspd_yscale:
        ax_nspd.set_yscale(nspd_yscale)
    if nspd_ylim:
        ax_nspd.set_ylim(nspd_ylim)
    ax_nspd.set_xlim(0, n_haplotypes)
    ax_nspd.set_xticks([])
    ax_nspd.set_ylabel('Shared haplotype length (bp)')

    # plot number of mutations
    ##########################

    ax_muts = fig.add_subplot(gs[1])
    sns.despine(ax=ax_muts, offset=5, bottom=True)
    x = np.arange(0, n_haplotypes - 1) + .5
    y = muts
    ax_muts.bar(x, y, width=1, align='edge', color='k')
    ax_muts.set_xlim(0, n_haplotypes)
    ax_muts.set_xticks([])
    if muts_ylim:
        ax_muts.set_ylim(*muts_ylim)
    ax_muts.set_ylabel('No. mutations')
    ax_muts.grid(axis='y')

    # plot t_hat
    ############

    ax_that = fig.add_subplot(gs[2])
    sns.despine(ax=ax_that, offset=5, bottom=True)
    x = np.arange(0, n_haplotypes - 1) + .5
    if nspd_accessible is not None:
        d = nspd_accessible
    else:
        d = nspd
    that = (1 + muts) / (2 * ((d * rr) + (d * mu)))
    ax_that.bar(x, that, width=1, align='edge', color='k')
    ax_that.set_xlim(0, n_haplotypes)
    ax_that.set_xticks([])
    ax_that.set_yscale('log')
    if that_ylim:
        ax_that.set_ylim(*that_ylim)
    ax_that.set_ylabel('$\hat{t}$', rotation=0, ha='right', va='center')
    ax_that.grid(axis='y')

    # plot haplotype colors
    #######################

    ax_hcol = fig.add_subplot(gs[3])
    sns.despine(ax=ax_hcol, left=True, bottom=True)
    ax_hcol.broken_barh([(i, 1) for i in range(n_haplotypes)], yrange=(0, 1),
                        color=pop_colors)
    ax_hcol.set_xlim(0, n_haplotypes)
    ax_hcol.set_xticks([])
    ax_hcol.set_yticks([])
    ax_hcol.set_ylabel('Population', rotation=0, ha='right', va='center')

    # plot display haplotypes
    #########################

    ax = fig.add_subplot(gs[4])
    plot_haplotypes(ax, haps_display, haps_display_vlbl)
    ax.set_xlim(0, n_haplotypes)
    ax.set_xlabel('Haplotypes')

    # final tidy up
    ###############

    gs.tight_layout(fig, h_pad=0)


def _shared_distance(spl, dist, jitter=False):
    spd = np.empty_like(spl, dtype='i4')
    n = dist.shape[0]
    for i in range(spl.shape[0]):
        l = spl[i]
        if l >= n:
            # homozygosity extends beyond end of data
            d = dist[-1]
        else:
            max_dist = dist[l]
            if jitter:
                if l > 0:
                    min_dist = dist[l-1]
                else:
                    min_dist = 1
                # use uniform approximation to estimate position of EHH break
                d = min_dist + (np.random.random() * (max_dist - min_dist))
            else:
                d = max_dist
        spd[i] = d
    return spd


def pairwise_haplotype_sharing(haps_ehh, haps_mut, dist_ehh, dist_mut, jitter=False):
    """Analyse sharing between all haplotype pairs."""

    haps_ehh = allel.HaplotypeArray(haps_ehh)
    haps_mut = allel.HaplotypeArray(haps_mut)
    n_haplotypes = haps_ehh.n_haplotypes
    assert n_haplotypes == haps_mut.n_haplotypes

    # compute length (no. variants) of shared prefix between pairs
    pspl = allel.opt.stats.pairwise_shared_prefix_lengths(
        np.asarray(haps_ehh, dtype='i1')
    )

    # compute length (physical distance) of shared prefix between neighbours
    pspd = _shared_distance(pspl, dist_ehh, jitter=jitter)

    # compute number of mutations on shared haplotypes
    muts = np.zeros_like(pspl)
    for i in range(n_haplotypes):
        for j in range(i+1, n_haplotypes):
            ix = allel.condensed_coords(i, j, n_haplotypes)
            # distance at which haplotypes diverge
            d = pspd[ix]
            # index into mutations array where haplotypes diverge
            idx_mut_div = bisect.bisect_right(dist_mut, d)
            # number of mutations
            m = np.count_nonzero(haps_mut[:idx_mut_div, i] != haps_mut[:idx_mut_div, j])
            muts[ix] = m

    return pspl, pspd, muts


# The cladogram function below works but not so well when distance is plotted on a logarithmic
# scale, where some lines can end up overlapping. Will leave commented out for now as a useful
# reference on how to make use of return values from scipy's dendrogram function.


# def cladogram(z, fill_threshold=0, leaf_height=0, leaf_width=10,
#               colors=None, default_color='k', ax=None, **kwargs):
#
#     if ax is None:
#         fig, ax = plt.subplots()
#
#     # compute the dendrogram
#     if colors is not None and 'link_color_func' not in kwargs:
#         if not isinstance(colors, np.ndarray):
#             colors = np.array(list(colors), dtype=object)
#         if colors.ndim == 1:
#             pass
#         elif colors.ndim == 2 and colors.shape[1] == 3:
#             # convert to hex for hashability
#             colors = np.array([mpl.colors.rgb2hex(tuple(c)) for c in colors])
#         else:
#             raise TypeError('bad colors')
#
#         tree = scipy.cluster.hierarchy.to_tree(z)
#
#         def link_color_func(k):
#             leaves = tree.get_descendant(k).pre_order()
#             leaf_colors = set(colors[leaves])
#             if len(leaf_colors) > 1:
#                 return default_color
#             else:
#                 return colors[leaves[0]]
#
#         kwargs['link_color_func'] = link_color_func
#
#     # get args
#     linewidth = kwargs.pop('lw', None)
#     if linewidth is None:
#         linewidth = kwargs.pop('linewidth', None)
#
#     r = scipy.cluster.hierarchy.dendrogram(z, no_plot=True, **kwargs)
#
#     # draw cladogram
#     for x, y, c in zip(r['icoord'], r['dcoord'], r['color_list']):
#         x1, x2, x3, x4 = x
#         y1, y2, y3, y4 = y
#         if y2 > fill_threshold:
#             ax.plot([x1, (x2 + x3)/2, x4], [y1, y2, y4], color=c, linewidth=linewidth)
#         else:
#             ax.add_patch(mpl.patches.Polygon([[x1, 0],
#                                               [x1, y1],
#                                               [(x2 + x3)/2, y2],
#                                               [x4, y4],
#                                               [x4, 0]],
#                                              color=c,
#                                              linewidth=linewidth,
#                                              zorder=-max(y)))
#
#     # draw leaves
#     if leaf_height > 0:
#         for i, l in enumerate(r['leaves']):
#             if colors is not None:
#                 c = colors[l]
#             else:
#                 c = default_color
#             ax.add_patch(plt.Rectangle((i * 10 + 5 - leaf_width/2, -leaf_height),
#                                        width=leaf_width,
#                                        height=leaf_height,
#                                        color=c,
#                                        linewidth=0))
#
#     # tidy up
#     ax.set_xlim(-5, len(r['leaves']) * 10 + 5)
#
#     return r


def cladogram(z, fill_threshold=0, default_color='k', colors=None, leaf_height=0,
              plot_leaf_func=None, count_sort=True, ax=None, plot_kws=None, fill_kws=None):
    """Plot a cladogram.

    Parameters
    ----------
    TODO

    """

    # setup axes
    if ax is None:
        _, ax = plt.subplots()

    # obtain a tree for convenience
    tree = scipy.cluster.hierarchy.to_tree(z)
    n = len(tree.pre_order())

    # normalize colors arg
    if colors is not None:
        if not isinstance(colors, np.ndarray):
            colors = np.array(list(colors), dtype=object)
        if colors.ndim == 1:
            pass
        elif colors.ndim == 2 and colors.shape[1] == 3:
            # convert to hex for hashability
            colors = np.array([mpl.colors.rgb2hex(tuple(c)) for c in colors])
        else:
            raise TypeError('bad colors')

    # start plotting
    if plot_kws is None:
        plot_kws = dict()
    if fill_kws is None:
        fill_kws = dict()
    if leaf_height > 0 and plot_leaf_func is None:
        plot_leaf_func = leaf_plotter_rect(leaf_height)
    _plot_clade(tree, offset=0, apex=None, fill_threshold=fill_threshold, ax=ax,
                plot_kws=plot_kws, fill_kws=fill_kws, default_color=default_color, colors=colors,
                plot_leaf_func=plot_leaf_func, count_sort=count_sort)

    # tidy up
    ax.set_xlim(0, n * 10)


def leaf_plotter_rect(height):

    def _plot(x, node, color, ax):
        ax.add_patch(plt.Rectangle((x - 5, -height),
                                   width=10,
                                   height=height,
                                   color=color,
                                   linewidth=0))

    return _plot


def leaf_plotter_marker(**kwargs):

    def _plot(x, node, color, ax):
        k = kwargs.copy()
        k.setdefault('color', color)
        ax.plot([x], [0], **k)

    return _plot


def _plot_clade(node, offset, apex, fill_threshold, ax, plot_kws, fill_kws, default_color,
                colors, plot_leaf_func, count_sort):

    if node.is_leaf():
        if plot_leaf_func is not None:
            if colors is not None:
                c = colors[node.id]
            else:
                c = default_color
            plot_leaf_func(offset + 5, node, c, ax)

    else:

        # count sort the children
        left, right = node.left, node.right
        if count_sort:
            if left.count > right.count:
                left, right = right, left

        # place the apex at the midpoint between the two child clades
        if apex is None:
            apex = offset + left.count * 10

        # figure out where to place the apex of each child clade
        left_offset = offset
        if left.is_leaf():
            left_apex = offset + 5
        else:
            left_apex = offset + min(left.left.count, left.right.count) * 10
        right_offset = offset + left.count * 10
        if right.is_leaf():
            right_apex = right_offset + 5
        else:
            right_apex = right_offset + min(right.left.count, right.right.count) * 10

        # figure out child colors
        if colors is None:
            left_color = default_color
            right_color = default_color
        else:
            left_colors = set(colors[left.pre_order()])
            if len(left_colors) > 1:
                left_color = default_color
            else:
                left_color = list(left_colors)[0]
            right_colors = set(colors[right.pre_order()])
            if len(right_colors) > 1:
                right_color = default_color
            else:
                right_color = list(right_colors)[0]

        # plot lines
        if node.dist > fill_threshold:
            x = [left_apex, apex]
            y = [left.dist, node.dist]
            ax.plot(x, y, color=left_color, **plot_kws)
            x = [right_apex, apex]
            y = [right.dist, node.dist]
            ax.plot(x, y, color=right_color, **plot_kws)

        # plot filled wedge
        else:
            x = [left_apex, apex]
            y = [left.dist, node.dist]
            ax.fill_between(x, 0, y, color=left_color, zorder=-node.dist, **fill_kws)
            x = [apex, right_apex]
            y = [node.dist, right.dist]
            ax.fill_between(x, 0, y, color=right_color, zorder=-node.dist, **fill_kws)

        # recurse
        kws = dict(fill_threshold=fill_threshold, ax=ax, plot_kws=plot_kws, fill_kws=fill_kws,
                   default_color=default_color, colors=colors, plot_leaf_func=plot_leaf_func,
                   count_sort=count_sort)
        _plot_clade(left, offset=left_offset, apex=left_apex, **kws)
        _plot_clade(right, offset=right_offset, apex=right_apex, **kws)
