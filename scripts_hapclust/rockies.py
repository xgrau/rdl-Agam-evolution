# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division
from bisect import bisect_left, bisect_right
import collections
import os


import numpy as np
# import matplotlib as mpl
import matplotlib.pyplot as plt
import lmfit
import seaborn as sns
palette = sns.color_palette()


def load_values(df, values_col, seqid, recmap, genome, spacing=0, seqid_col='seqid',
                start_col='start', end_col='end'):
    """Extract genome-wide selection scan values from a dataframe.

    Parameters
    ----------
    df : pandas.DataFrame
        A dataframe with windowed data. Expected to contain columns "chrom" (sequence
        identifier), "start", "stop", as well as `col`. Expect that "start" and "stop" are
        GFF-style 1-based stop-inclusive coords.
    values_col : str
        Name of column containing statistic values.
    seqid : str
        Sequence identifier. May also be a tuple, e.g., ('2R', '2L'), in which
        case the data for each sequence will be concatenated.
    recmap : dict [str -> array]
        Recombination map. A dictionary mapping sequence identifiers onto arrays,
        where each array holds the absolute recombination rate for each base
        in the sequence.
    genome : dict-like [str -> array]
        Genome sequences.
    spacing : int, optional
        Amount of physical distance to insert when concatenating data from
        multiple chromosome arms.
    seqid_col, start_col, end_col : str
        Column names for window sequence identifier, start and end coordinates.

    Returns
    -------
    starts
    ends
    gpos
    values

    """

    # handle multiple sequences
    if isinstance(seqid, (tuple, list)):
        assert len(seqid) == 2, 'can only concatenate two sequences'
        (starts1, ends1, gpos1, values1), (starts2, ends2, gpos2, values2) = \
            [load_values(df, values_col=values_col, seqid=c, recmap=recmap,
                         genome=genome, seqid_col=seqid_col, start_col=start_col,
                         end_col=end_col)
             for c in seqid]
        seq1_plen = len(genome[seqid[0]])
        seq1_glen = np.sum(recmap[seqid[0]])
        starts = np.concatenate([starts1, starts2 + seq1_plen + spacing])
        ends = np.concatenate([ends1, ends2 + seq1_plen + spacing])
        gpos = np.concatenate([gpos1, (gpos2 + seq1_glen +
                                       recmap[seqid[1]][0] * spacing)])
        values = np.concatenate([values1, values2])
        return starts, ends, gpos, values

    # extract data for single seqid
    df = df.reset_index().set_index(seqid_col)
    df_seq = df.loc[seqid]

    # extract window starts and ends
    starts = np.asarray(df_seq[start_col])
    ends = np.asarray(df_seq[end_col])

    # compute genetic length of each window
    glen = np.array([
        np.sum(recmap[seqid][int(start - 1):int(end)])
        for start, end in zip(starts, ends)
    ])

    # compute the genetic length position for each window
    gpos = np.cumsum(glen)

    # extract the values column
    values = np.asarray(df_seq[values_col])

    return starts, ends, gpos, values


def plot_values(df, values_col, seqid, recmap, spacing=0, start=None, end=None,
                figsize=(16, 3), distance='physical', ax=None, seqid_col='seqid',
                start_col='start', end_col='end'):
    """Convenience function to plot values from a windowed selection scan."""

    # extract data
    starts, ends, gpos, values = load_values(df, values_col, seqid, recmap,
                                             spacing=spacing, seqid_col=seqid_col,
                                             start_col=start_col, end_col=end_col)

    # determine x coord
    if distance == 'genetic':
        x = gpos
    else:
        x = (starts + ends) / 2

    if start is not None or end is not None:
        flt = np.ones(x.shape, dtype=bool)
        if start is not None:
            flt = flt & (x >= start)
        if end is not None:
            flt = flt & (x <= end)
        x = x[flt]
        values = values[flt]

    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    ax.plot(x, values, linestyle=' ', marker='o', mfc='none', markersize=3,
            color=palette[0], mew=1)
    ax.set_title(values_col)
    if distance == 'genetic':
        ax.set_xlabel('Sequence {} position (cM)'.format(seqid))
    else:
        ax.set_xlabel('Sequence {} position (bp)'.format(seqid))
    if fig is not None:
        fig.tight_layout()
    return ax


def exponential(x, amplitude, decay, c, cap):
    """Exponential decay function.

    Parameters
    ----------
    x : ndarray
        Independent variable.
    amplitude : float
        Amplitude parameter.
    decay : float
        Decay parameter.
    c : float
        Constant baseline.
    cap : float
        Maximum value that the result can take.

    Returns
    -------
    y : ndarray

    """

    # compute exponential
    y = c + amplitude * np.exp(-x / decay)

    # apply cap
    y = y.clip(None, cap)

    return y


def symmetric_exponential_peak(x, center, amplitude, decay, c, cap):
    """Symmetric exponential decay peak function.

    Parameters
    ----------
    x : ndarray
        Independent variable.
    center : int or float
        The center of the peak.
    amplitude : float
        Amplitude parameter.
    decay : float
        Decay parameter.
    c : float
        Constant baseline.
    cap : float
        Maximum value that the result can take.

    Returns
    -------
    y : ndarray

    """

    # locate the center
    ix_cen = bisect_right(x, center)

    # compute left flank
    xl = center - x[:ix_cen]
    yl = c + amplitude * np.exp(-xl / decay)

    # compute right flank
    xr = x[ix_cen:] - center
    yr = c + amplitude * np.exp(-xr / decay)

    # prepare output
    y = np.concatenate([yl, yr])

    # apply cap
    y = y.clip(None, cap)

    return y


def asymmetric_exponential_peak(x, center, amplitude, left_decay, right_decay,
                                c, cap):
    """Asymmetric exponential decay peak function.

    Parameters
    ----------
    x : ndarray
        Independent variable.
    center : int or float
        The center of the peak.
    amplitude : float
        Amplitude parameter.
    left_decay : float
        Decay for left-hand flank.
    right_decay : float
        Decay for right-hand flank.
    c : float
        Constant baseline.
    cap : float
        Maximum value that the result can take.

    Returns
    -------
    y : ndarray

    """

    # locate the center
    ix_cen = bisect_right(x, center)

    # compute left flank
    xl = center - x[:ix_cen]
    yl = c + amplitude * np.exp(-xl / left_decay)

    # compute right flank
    xr = x[ix_cen:] - center
    yr = c + amplitude * np.exp(-xr / right_decay)

    # prepare output
    y = np.concatenate([yl, yr])

    # apply cap
    y = y.clip(None, cap)

    return y


FitResult = collections.namedtuple(
    'FitResult',
    'delta_aic min_delta_aic sum_delta_aic peak_result null_result loc xx yy '
    'best_fit peak residual peak_start_ix peak_end_ix peak_start_x '
    'peak_end_x baseline baseline_stderr'
)


def find_peak_limits(best_fit, baseline, stderr):
    ix_peak_start = ix_peak_end = None
    # work forward to find peak start
    for i in range(best_fit.shape[0]):
        v = best_fit[i]
        if ix_peak_start is None:
            if v > baseline + 1 * stderr:
                ix_peak_start = i
    # work backwards to find peak end
    for i in range(best_fit.shape[0] - 1, -1, -1):
        v = best_fit[i]
        if ix_peak_end is None:
            if v > baseline + 1 * stderr:
                ix_peak_end = i
                break
    return ix_peak_start, ix_peak_end


# noinspection PyUnresolvedReferences
class PeakFitter(object):
    """Abstract base class for peak fitters."""

    def fit(self, x, y, center, flank):

        # slice out the region of data to fit against
        ix_left = bisect_left(x, center - flank)
        ix_right = bisect_right(x, center + flank)
        loc = slice(ix_left, ix_right)
        xx = x[loc] - center  # make relative to center
        yy = y[loc]

        # fit the null model - remove one outlier
        no_outlier = yy < yy.max()
        null_result = self.null_model.fit(yy[no_outlier], x=xx[no_outlier],
                                          params=self.null_params)

        # fit the peak model
        peak_result = self.peak_model.fit(yy, x=xx, params=self.peak_params)

        # obtain difference in AIC
        delta_aic = null_result.aic - peak_result.aic

        # obtain best fit for peak data for subtracting from values
        baseline = peak_result.params['c'].value
        baseline_stderr = peak_result.params['c'].stderr
        best_fit = peak_result.best_fit
        peak = best_fit - baseline
        residual = yy - peak

        # figure out the width of the peak
        peak_start_ix = peak_end_ix = peak_start_x = peak_end_x = None
        rix_peak_start, rix_peak_end = find_peak_limits(best_fit, baseline,
                                                         baseline_stderr)
        if rix_peak_start is not None and rix_peak_end is not None:
            peak_start_ix = ix_left + rix_peak_start
            peak_end_ix = ix_left + rix_peak_end
            peak_start_x = xx[rix_peak_start]
            peak_end_x = xx[rix_peak_end]

        return FitResult(delta_aic, delta_aic, delta_aic, peak_result,
                         null_result, loc, xx, yy, best_fit, peak, residual,
                         peak_start_ix, peak_end_ix, peak_start_x,
                         peak_end_x, baseline, baseline_stderr)


class GaussianPeakFitter(PeakFitter):

    def __init__(self, amplitude, sigma, c):

        # initialise null model
        null_model = lmfit.models.ConstantModel()
        null_params = lmfit.Parameters()
        null_params['c'] = c
        self.null_model = null_model
        self.null_params = null_params

        # initialise peak model
        peak_model = lmfit.models.GaussianModel() + lmfit.models.ConstantModel()
        peak_params = lmfit.Parameters()
        peak_params['center'] = lmfit.Parameter(value=0, vary=False)
        peak_params['amplitude'] = amplitude
        peak_params['sigma'] = sigma
        peak_params['c'] = c
        self.peak_model = peak_model
        self.peak_params = peak_params


class LorentzianPeakFitter(PeakFitter):

    def __init__(self, amplitude, sigma, c):

        # initialise null model
        null_model = lmfit.models.ConstantModel()
        null_params = lmfit.Parameters()
        null_params['c'] = c
        self.null_model = null_model
        self.null_params = null_params

        # initialise peak model
        peak_model = (lmfit.models.LorentzianModel() +
                      lmfit.models.ConstantModel())
        peak_params = lmfit.Parameters()
        peak_params['center'] = lmfit.Parameter(value=0, vary=False)
        peak_params['amplitude'] = amplitude
        peak_params['sigma'] = sigma
        peak_params['c'] = c
        self.peak_model = peak_model
        self.peak_params = peak_params


class SymmetricExponentialPeakFitter(PeakFitter):

    def __init__(self, amplitude, decay, c, cap):

        # initialise null model
        null_model = lmfit.models.ConstantModel()
        null_params = lmfit.Parameters()
        null_params['c'] = c
        self.null_model = null_model
        self.null_params = null_params

        # initialise peak model
        peak_model = lmfit.Model(symmetric_exponential_peak)
        peak_params = lmfit.Parameters()
        peak_params['center'] = lmfit.Parameter(value=0, vary=False)
        peak_params['amplitude'] = amplitude
        peak_params['decay'] = decay
        peak_params['c'] = c
        peak_params['cap'] = cap
        self.peak_model = peak_model
        self.peak_params = peak_params


class AsymmetricExponentialPeakFitter(PeakFitter):

    def __init__(self, amplitude, left_decay, right_decay, c, cap):

        # initialise null model
        null_model = lmfit.models.ConstantModel()
        null_params = lmfit.Parameters()
        null_params['c'] = c
        self.null_model = null_model
        self.null_params = null_params

        # initialise peak model
        peak_model = lmfit.Model(asymmetric_exponential_peak)
        peak_params = lmfit.Parameters()
        peak_params['center'] = lmfit.Parameter(value=0, vary=False)
        peak_params['amplitude'] = amplitude
        peak_params['left_decay'] = left_decay
        peak_params['right_decay'] = right_decay
        peak_params['c'] = c
        peak_params['cap'] = cap
        self.peak_model = peak_model
        self.peak_params = peak_params


class PairExponentialPeakFitter(PeakFitter):

    def __init__(self, amplitude, decay, c, cap):

        # initialise null model
        null_model = lmfit.models.ConstantModel()
        null_params = lmfit.Parameters()
        # allow this one to vary freely
        null_params['c'] = lmfit.Parameter(value=c.value, vary=True)
        self.null_model = null_model
        self.null_params = null_params

        # initialise peak model
        peak_model = lmfit.Model(exponential)
        peak_params = lmfit.Parameters()
        peak_params['amplitude'] = amplitude
        peak_params['decay'] = decay
        peak_params['c'] = c
        peak_params['cap'] = cap
        self.peak_model = peak_model
        self.peak_params = peak_params

    def fit(self, x, y, center, flank):

        # slice out the region of data to fit against
        ix_left = bisect_left(x, center - flank)
        ix_right = bisect_right(x, center + flank)
        loc = slice(ix_left, ix_right)
        xx = x[loc] - center
        yy = y[loc]

        # split into left and right flanks
        ix_center = bisect_right(xx, 0)
        xl = -xx[:ix_center]
        yl = yy[:ix_center]
        xr = xx[ix_center:]
        yr = yy[ix_center:]

        # default outputs
        delta_aic = min_delta_aic = sum_delta_aic = peak_result = \
            null_result = best_fit = peak = residual = peak_start_ix = \
            peak_end_ix = peak_start_x = peak_end_x = baseline = \
            baseline_stderr = None

        # check there's some data on both flanks
        if xl.shape[0] > 3 and xr.shape[0] > 3:
            # fit each flank separately

            # find outliers
            no_outlier_l = yl < yl.max()
            no_outlier_r = yr < yr.max()

            # check there's data after excluding outliers
            if (np.count_nonzero(no_outlier_l) > 0 and
                    np.count_nonzero(no_outlier_r) > 0):

                # fit the null model - allow one outlier
                null_result_l = self.null_model.fit(yl[no_outlier_l],
                                                    x=xl[no_outlier_l],
                                                    params=self.null_params)
                null_result_r = self.null_model.fit(yr[no_outlier_r],
                                                    x=xr[no_outlier_r],
                                                    params=self.null_params)

                # fit the peak model
                peak_result_l = self.peak_model.fit(yl, x=xl,
                                                    params=self.peak_params)
                peak_result_r = self.peak_model.fit(yr, x=xr,
                                                    params=self.peak_params)

                # obtain difference in AIC
                delta_aic_l = null_result_l.aic - peak_result_l.aic
                delta_aic_r = null_result_r.aic - peak_result_r.aic
                delta_aic = delta_aic_l, delta_aic_r
                min_delta_aic = min(delta_aic)
                sum_delta_aic = sum(delta_aic)

                # determine baseline
                baseline_l = peak_result_l.params['c'].value
                baseline_r = peak_result_r.params['c'].value
                baseline_stderr_l = peak_result_l.params['c'].stderr
                baseline_stderr_r = peak_result_r.params['c'].stderr
                baseline = max([baseline_l, baseline_r])
                baseline_stderr = max([baseline_stderr_l, baseline_stderr_r])

                # obtain best fit for peak data for subtracting from values
                best_fit_l = peak_result_l.best_fit
                peak_l = (best_fit_l - baseline).clip(0, None)
                residual_l = yl - peak_l
                best_fit_r = peak_result_r.best_fit
                peak_r = (best_fit_r - baseline).clip(0, None)
                residual_r = yr - peak_r

                # prepare output
                peak_result = peak_result_l, peak_result_r
                null_result = null_result_l, null_result_r
                # noinspection PyUnresolvedReferences
                best_fit = np.concatenate([best_fit_l, best_fit_r])
                # noinspection PyUnresolvedReferences
                peak = np.concatenate([peak_l, peak_r])
                # noinspection PyUnresolvedReferences
                residual = np.concatenate([residual_l, residual_r])

                # figure out the width of the peak
                peak_start_ix = peak_end_ix = peak_start_x = peak_end_x = None
                rix_peak_start, rix_peak_end = find_peak_limits(best_fit,
                                                                 baseline,
                                                                 baseline_stderr)
                if rix_peak_start is not None and rix_peak_end is not None:
                    peak_start_ix = ix_left + rix_peak_start
                    peak_end_ix = ix_left + rix_peak_end
                    peak_start_x = xx[rix_peak_start]
                    peak_end_x = xx[rix_peak_end]

        return FitResult(delta_aic, min_delta_aic, sum_delta_aic, peak_result,
                         null_result, loc, xx, yy, best_fit, peak, residual,
                         peak_start_ix, peak_end_ix, peak_start_x,
                         peak_end_x, baseline, baseline_stderr)


def scan_fit(x, y, flank, fitter, centers, delta_aics, fits,
             start_index, stop_index, log):

    # N.B., there may be more centers than data points, because nans must
    # have been removed from data, but we will fit at all windows
    # noinspection PyUnresolvedReferences
    assert not np.any(np.isnan(y))
    assert x.shape == y.shape
    assert (centers.shape[0] == delta_aics.shape[0] == fits.shape[0])

    # determine region to scan over
    n = centers.shape[0]
    if start_index is None:
        start_index = 0
    if stop_index is None:
        stop_index = n
    assert start_index >= 0
    assert stop_index <= n

    # iterate and fit
    for i in range(start_index, stop_index):
        if i % 100 == 0:
            log('scan progress', i, centers[i])

        # central position to fit at
        center = centers[i]

        # fit the peak
        fit = fitter.fit(x, y, center, flank)

        # store the results
        fits[i] = fit
        if fit.delta_aic is not None:
            delta_aics[i] = fit.delta_aic


Peak = collections.namedtuple(
    'Peak',
    'best_fit minor_delta_aic sum_delta_aic best_ix epicenter_start '
    'epicenter_end focus_start focus_end peak_start peak_end ppos values '
    'delta_aic'
)


def find_best_peak(delta_aics, min_minor_delta_aic, in_peak):
    if delta_aics.ndim == 2:
        # make sure we don't find peaks where the minor di is below required
        # threshold, or where we've found peaks before
        di = delta_aics.copy()
        di[(di.min(axis=1) < min_minor_delta_aic) | in_peak] = 0
        best_ix = np.argmax(di.sum(axis=1))
        minor_delta_aic = float(min(delta_aics[best_ix]))
        sum_delta_aic = float(sum(delta_aics[best_ix]))
    else:
        # make sure we don't find peaks where we've found peaks before
        di = delta_aics.copy()
        di[in_peak] = 0
        best_ix = np.argmax(di)
        minor_delta_aic = float(delta_aics[best_ix])
        sum_delta_aic = float(delta_aics[best_ix])
    return best_ix, minor_delta_aic, sum_delta_aic


def find_peaks(window_starts, window_ends, gpos, values, flank, fitter,
               min_minor_delta_aic=30, min_total_delta_aic=60, max_iter=20,
               extend_delta_aic_frc=0.05, verbose=True, output_dir=None,
               log_file='log.txt'):
    """DOC ME"""

    log_path = None
    if log_file:
        if output_dir:
            log_path = os.path.join(output_dir, log_file)
        else:
            log_path = log_file
        if os.path.exists(log_path):
            os.remove(log_path)

    def log(*args):
        if verbose:
            if log_path:
                with open(log_path, mode='a') as f:
                    kwargs = dict(file=f)
                    print(*args, **kwargs)
            else:
                print(*args)

    window_starts = np.asarray(window_starts)
    window_ends = np.asarray(window_ends)
    gpos = np.asarray(gpos)
    values = np.asarray(values)
    assert (window_starts.shape == window_ends.shape == gpos.shape ==
            values.shape)
    n = gpos.shape[0]

    # setup working data structures
    if isinstance(fitter, PairExponentialPeakFitter):
        delta_aics = np.zeros((n, 2), dtype='f8')
    else:
        delta_aics = np.zeros(n, dtype='f8')
    log('delta_aics', delta_aics.shape)
    fits = np.empty(n, dtype=object)
    in_peak = np.zeros(n, dtype=bool)

    # strip out missing data
    # noinspection PyUnresolvedReferences
    missing = np.isnan(values)
    x = gpos[~missing]
    y = values[~missing]
    starts_nomiss = window_starts[~missing]
    ends_nomiss = window_ends[~missing]
    ppos_nomiss = (starts_nomiss + ends_nomiss) / 2
    values_nomiss = values[~missing]

    # first pass model fits
    scan_fit(x, y, flank=flank, fitter=fitter, centers=gpos,
             delta_aics=delta_aics, fits=fits, log=log, start_index=None,
             stop_index=None)

    # keep track of which iteration we're on, starting from 1 (be human
    # friendly)
    iteration = 1

    # find the first peak
    best_ix, minor_delta_aic, sum_delta_aic = find_best_peak(
        delta_aics=delta_aics, min_minor_delta_aic=min_minor_delta_aic,
        in_peak=in_peak)
    best_fit = fits[best_ix]
    log('first pass', best_ix, minor_delta_aic)

    while sum_delta_aic > min_total_delta_aic and iteration < max_iter:

        log('*' * 80)
        log('Iteration', iteration)
        log('Peak index:', best_ix)
        log('Delta AIC:', delta_aics[best_ix])
        log('Window:', window_starts[best_ix], window_ends[best_ix])
        log('*' * 80)

        iter_out_dir = None
        if minor_delta_aic < min_minor_delta_aic:
            skip = True
            log('SKIPPING: POOR FLANK')
        else:
            skip = False
            if output_dir:
                iter_out_dir = os.path.join(output_dir, str(iteration))
                os.makedirs(iter_out_dir, exist_ok=True)

        plot_peak_context(x, y, gpos=gpos, values=values,
                          delta_aics=delta_aics,
                          best_ix=best_ix, iter_out_dir=iter_out_dir,
                          iteration=iteration)
        plot_peak_fit(best_fit, iter_out_dir=iter_out_dir)

        log('find extent of region under selection')
        epicenter_start = int(window_starts[best_ix])
        focus_start = epicenter_start
        # N.B., center is included in left flank, so we'll include the next
        # window along within the hit
        epicenter_end = int(window_ends[best_ix+1])
        focus_end = epicenter_end
        # search left
        i = best_ix - 1
        while 0 <= i < n:
            if fits[i].sum_delta_aic is None:
                break
            if ((sum_delta_aic - fits[i].sum_delta_aic) <
                    (extend_delta_aic_frc * sum_delta_aic)):
                log('extend hit left', fits[i].sum_delta_aic)
                focus_start = int(window_starts[i])
                i -= 1
            else:
                break
        # search right
        i = best_ix + 1
        while 0 <= i < n:
            if fits[i].sum_delta_aic is None:
                break
            if ((sum_delta_aic - fits[i].sum_delta_aic) <
                    (extend_delta_aic_frc * sum_delta_aic)):
                log('extend hit right', fits[i].sum_delta_aic)
                focus_end = int(window_ends[i+1])
                i += 1
            else:
                break

        log('find flanking region')
        peak_start = peak_end = None
        if best_fit.peak_start_ix is not None:
            peak_start = int(starts_nomiss[best_fit.peak_start_ix])
        if best_fit.peak_end_ix is not None:
            peak_end = int(ends_nomiss[best_fit.peak_end_ix])

        plot_peak_location(best_ix=best_ix, best_fit=best_fit,
                           focus_start=focus_start,
                           focus_end=focus_end,
                           window_starts=window_starts,
                           window_ends=window_ends,
                           starts_nomiss=starts_nomiss,
                           ends_nomiss=ends_nomiss,
                           ppos_nomiss=ppos_nomiss,
                           values_nomiss=values_nomiss,
                           iter_out_dir=iter_out_dir)
        plot_peak_targetting(best_ix=best_ix, best_fit=best_fit,
                             focus_start=focus_start,
                             focus_end=focus_end,
                             window_starts=window_starts,
                             window_ends=window_ends,
                             starts_nomiss=starts_nomiss,
                             ends_nomiss=ends_nomiss,
                             iter_out_dir=iter_out_dir,
                             delta_aics=delta_aics)

        # filter out hits with too much parameter uncertainty
        # params_check = True
        # for k in best_fit.peak_result.params:
        #     p = best_fit.peak_result.params[k]
        #     if (p.stderr / p.value) > max_param_stderr:
        #         params_check = False
        #         log('failed param check: ', p)
        #
        # if params_check:
        if not skip:
            yield Peak(best_fit, minor_delta_aic, sum_delta_aic, best_ix,
                       epicenter_start, epicenter_end, focus_start, focus_end,
                       peak_start, peak_end, ppos_nomiss[best_fit.loc],
                       values_nomiss[best_fit.loc], delta_aics[best_ix])
            iteration += 1

        # subtract peak from values
        y[best_fit.loc] = (y[best_fit.loc] - best_fit.peak).clip(0, None)

        # rescan region around the peak
        center = gpos[best_ix]
        ix_rescan_left = bisect_left(gpos, center - (flank * 2))
        ix_rescan_right = bisect_right(gpos, center + (flank * 2))
        log('rescan', ix_rescan_left, ix_rescan_right)
        scan_fit(x, y, flank=flank, fitter=fitter, centers=gpos,
                 delta_aics=delta_aics, fits=fits,
                 start_index=ix_rescan_left, stop_index=ix_rescan_right,
                 log=log)

        # keep track of where we have found peaks, so we don't find something
        # in the residuals
        ix_peak_start = bisect_left(window_starts, peak_start)
        ix_peak_end = bisect_right(window_ends, peak_end)
        in_peak[ix_peak_start:ix_peak_end] = True

        # find next peak
        best_ix, minor_delta_aic, sum_delta_aic = find_best_peak(
            delta_aics=delta_aics, min_minor_delta_aic=min_minor_delta_aic,
            in_peak=in_peak)
        best_fit = fits[best_ix]
        log('next peak:', best_ix, delta_aics[best_ix])

    log('all done')


def plot_peak_fit(fit, figsize=(8, 2.5), iter_out_dir=None,
                  xlabel='Genetic distance (cM)', ylabel='Selection statistic'):
    # noinspection PyTypeChecker
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=figsize, facecolor='w')

    ax = axs[0]
    # plot width of the peak
    if fit.peak_start_x and fit.peak_end_x:
        ax.axvspan(fit.peak_start_x, fit.peak_end_x, facecolor=palette[0],
                   alpha=.2)
    # ax.axvline(0, color='k', lw=.5, linestyle='--')
    # plot the fit
    ax.plot(fit.xx, fit.best_fit, lw=.5, linestyle='--', color='k')
    # # plot the baseline
    # if fit.baseline is not None:
    #     ax.axhline(fit.baseline, lw=1, linestyle='--', color='k')
    # plot the data
    ax.plot(fit.xx, fit.yy, marker='o', linestyle=' ', markersize=3,
            mfc='none', color=palette[0], mew=.5)
    if isinstance(fit.delta_aic, (list, tuple)):
        ax.text(.02, .98, r'$\Delta_{i}$ : %.1f' % fit.delta_aic[0],
                transform=ax.transAxes, ha='left', va='top')
        ax.text(.98, .98, r'$\Delta_{i}$ : %.1f' % fit.delta_aic[1],
                transform=ax.transAxes, ha='right', va='top')
    else:
        ax.text(.02, .98, r'$\Delta_{i}$ : %.1f' % fit.delta_aic,
                transform=ax.transAxes, ha='left', va='top')
    ax.set_title('Peak fit')
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.set_ylim(bottom=0)

    ax = axs[1]
    if fit.baseline is not None:
        ax.axhline(fit.baseline, lw=.5, linestyle='--', color='k')
    ax.plot(fit.xx, fit.residual, marker='o', linestyle=' ', markersize=3,
            mfc='none', color=palette[0], mew=.5)
    ax.set_ylim(axs[0].get_ylim())
    ax.set_title('Residual')
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.set_ylim(bottom=0)

    ax = axs[2]
    ax.hist(fit.residual, bins=30)
    # if fit.baseline is not None:
    #     ax.axvline(fit.baseline, lw=.5, linestyle='--', color='k')
    ax.set_xlabel(ylabel)
    ax.set_ylabel('Frequency')
    ax.set_title('Residual')

    fig.tight_layout()

    if iter_out_dir:
        fig.savefig(os.path.join(iter_out_dir, 'peak_fit.png'),
                    bbox_inches='tight', dpi=150, facecolor='w')
        plt.close()


def plot_peak_location(best_ix, best_fit, focus_start, focus_end, window_starts,
                       window_ends, starts_nomiss, ends_nomiss, ppos_nomiss,
                       values_nomiss, iter_out_dir, figsize=(6, 3)):

    fig, ax = plt.subplots(figsize=figsize)

    # plot region under peak
    if (best_fit.peak_start_ix is not None and
            best_fit.peak_end_ix is not None):
        ax.axvspan(starts_nomiss[best_fit.peak_start_ix] / 1e6, focus_start / 1e6,
                   color=palette[0], alpha=.2)
        ax.axvspan(focus_end / 1e6, ends_nomiss[best_fit.peak_end_ix] / 1e6,
                   color=palette[0], alpha=.2)

    # plot hit region
    ax.axvspan(focus_start / 1e6, focus_end / 1e6, color=palette[3], alpha=.3)

    # plot best window
    ax.axvspan(window_starts[best_ix] / 1e6, window_ends[best_ix+1] / 1e6,
               color=palette[3], alpha=.3)

    # plot fit
    ax.plot(ppos_nomiss[best_fit.loc] / 1e6, best_fit.best_fit,
            linestyle='--', color='k', lw=.5)

    # plot data
    ax.plot(ppos_nomiss[best_fit.loc] / 1e6, values_nomiss[best_fit.loc], marker='o',
            linestyle=' ', color=palette[0], mew=1, mfc='none',
            markersize=3)

    # tidy up
    ax.set_xlabel('Position (Mbp)')
    ax.set_ylabel('Selection statistic')
    ax.set_ylim(bottom=0)
    fig.tight_layout()

    if iter_out_dir:
        fig.savefig(os.path.join(iter_out_dir, 'peak_location.png'),
                    bbox_inches='tight', dpi=150, facecolor='w')
        plt.close()


def plot_peak_targetting(best_ix, best_fit, focus_start, focus_end, window_starts,
                         window_ends, delta_aics, starts_nomiss, ends_nomiss,
                         iter_out_dir, figsize=(8, 3.5)):

    fig, ax = plt.subplots(figsize=figsize)

    # plot region under peak
    if (best_fit.peak_start_ix is not None and
            best_fit.peak_end_ix is not None):
        ax.axvspan(starts_nomiss[best_fit.peak_start_ix] / 1e6, focus_start / 1e6,
                   color=palette[0], alpha=.2)
        ax.axvspan(focus_end / 1e6, ends_nomiss[best_fit.peak_end_ix] / 1e6,
                   color=palette[0], alpha=.2)

    # plot hit region
    ax.axvspan(focus_start / 1e6, focus_end / 1e6, color=palette[3], alpha=.3)

    # plot best window
    ax.axvspan(window_starts[best_ix] / 1e6, window_ends[best_ix+1] / 1e6,
               color=palette[3], alpha=.3)

    # plot delta AICs
    if best_fit.peak_start_ix is not None:
        peak_start = starts_nomiss[best_fit.peak_start_ix]
    else:
        peak_start = 0
    if best_fit.peak_end_ix is not None:
        peak_end = ends_nomiss[best_fit.peak_end_ix]
    else:
        peak_end = window_ends[-1]
    di_loc = slice(bisect_left(window_starts, peak_start),
                   bisect_right(window_ends, peak_end))
    x = window_ends[di_loc] / 1e6
    if delta_aics.ndim == 2:
        y1 = delta_aics[di_loc, 0]
        y2 = delta_aics[di_loc, 1]
        y3 = delta_aics[di_loc, :].sum(axis=1)
        ax.plot(x, y1, lw=2, label='left flank')
        ax.plot(x, y2, lw=2, label='right flank')
        ax.plot(x, y3, lw=2, label='flanks combined')
        ax.legend(loc='upper right')
    else:
        y1 = delta_aics[di_loc]
        ax.plot(x, y1, lw=2)

    # tidy up
    ax.set_xlabel('Position (Mbp)')
    ax.set_ylabel(r'$\Delta_{i}$')
    ax.set_ylim(bottom=0)
    ax.set_title('Peak targetting')
    fig.tight_layout()

    if iter_out_dir:
        fig.savefig(os.path.join(iter_out_dir, 'peak_targetting.png'),
                    bbox_inches='tight', dpi=150, facecolor='w')
        plt.close()


def plot_peak_context(x, y, gpos, values, delta_aics, best_ix,
                      iter_out_dir, iteration):
    # noinspection PyTypeChecker
    fig, axs = plt.subplots(nrows=3, figsize=(8, 5), sharex=True)

    ax = axs[0]
    ax.set_ylim(bottom=0)
    ax.axvline(gpos[best_ix], linestyle='--', lw=.5, color='k')
    ax.plot(gpos, values, marker='o', linestyle=' ', markersize=2,
            mfc='none', mew=.5)
    ax.set_ylabel('Selection statistic')
    ax.set_title('Original values')

    ax = axs[1]
    ax.set_ylim(bottom=0)
    ax.axvline(gpos[best_ix], linestyle='--', lw=.5, color='k')
    ax.plot(x, y, marker='o', linestyle=' ', markersize=2,
            mfc='none', mew=.5)
    ax.set_ylabel('Selection statistic')
    ax.set_title('Remaining values at iteration {}'.format(iteration))

    ax = axs[2]
    if delta_aics.ndim == 2:
        di = delta_aics.sum(axis=1)
    else:
        di = delta_aics
    ax.plot(gpos, di, lw=.5)
    ax.text(gpos[best_ix], di[best_ix], 'v', va='bottom', ha='center')
    ax.set_ylim(bottom=0)
    ax.set_ylabel(r'$\Delta_{i}$')
    ax.set_xlabel('Position (cM)')
    ax.set_title('Peak model fit at iteration {}'.format(iteration))

    fig.tight_layout()
    if iter_out_dir:
        fig.savefig(os.path.join(iter_out_dir, 'peak_context.png'),
                    bbox_inches='tight', dpi=150, facecolor='w')
        plt.close()
