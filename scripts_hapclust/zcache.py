# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division
import itertools
import operator
import hashlib
import pickle
import sys


import zarr
import numpy as np


def log(*msg):
    """Simple logging function that flushes immediately to stdout."""
    s = ' '.join(map(str, msg))
    print(s, file=sys.stdout)
    sys.stdout.flush()


# this object is used to delineate the beginning of keyword arguments
kwarg_separator = '__kwargs__'


# this key is used in attributes to indicate successful storage
success_key = '__ok__'


# this key is used in attributes to indicate a scalar value
scalar_key = '__is_scalar__'


def linearize_kwargs(*args, **kwargs):
    """Function to convert function call arguments into a simple sequence."""
    if kwargs:
        args = (
            args +
            (kwarg_separator,) +
            tuple(itertools.chain(*sorted(kwargs.items(), key=operator.itemgetter(0))))
        )
    return args


class PickleKeySmith(object):
    """Class for generating keys using the pickle protocol."""

    def __init__(self, protocol=2, fix_imports=True, hash_name='md5'):
        self.protocol = protocol
        self.fix_imports = fix_imports
        self.hash_name = hash_name

    def cut(self, *args, **kwargs):
        """Cut a key for the given arguments."""
        objects = linearize_kwargs(*args, **kwargs)
        h = hashlib.new(self.hash_name)
        for obj in objects:
            s = pickle.dumps(obj, protocol=self.protocol, fix_imports=self.fix_imports)
            h.update(s)
        return h.hexdigest()


class CacheMiss(Exception):

    def __init__(self, namespace, *args, **kwargs):
        super(CacheMiss, self).__init__(namespace, args, kwargs)


class Cache(object):
    """Cache class.

    Parameters
    ----------
    backend : zarr or h5py group, optional
        Group to store results in. If not provided, a zarr in-memory group will be used.
    key_smith : KeySmith
        Key smith. If not provided, a PickleKeySmith with default parameters will be used.
    dataset_kwargs : dict-like
        Keyword arguments passed through to backend.create_dataset() when storing results in the
        cache.

    """

    def __init__(self, backend=None, key_smith=None, dataset_kwargs=None, verbose=True):
        if backend is None:
            backend = zarr.group()
        self.backend = backend
        if key_smith is None:
            key_smith = PickleKeySmith()
        self.key_smith = key_smith
        if dataset_kwargs is None:
            dataset_kwargs = dict(chunks=True)
        self.dataset_kwargs = dataset_kwargs
        self.verbose = verbose

    def get(self, namespace, *args, **kwargs):
        """Retrieve results from the cache."""

        try:

            # obtain group holding cached results for the given namespace
            namespace_grp = self.backend[namespace]

            # obtain a key for the given arguments
            key = self.key_smith.cut(*args, **kwargs)

            # obtain group holding cached results for the given key
            result_grp = namespace_grp[key]

            # check storage consistency
            check = result_grp.attrs[success_key]

        except KeyError:
            if self.verbose:
                log('[%s] cache miss' % namespace, args, kwargs)
            raise CacheMiss(namespace, *args, **kwargs)

        else:
            if self.verbose:
                log('[%s] cache hit' % namespace, args, kwargs)

            # proceed with retrieving cached results
            names = sorted(result_grp.keys())
            ret = [result_grp[n] for n in names]

            # handle scalars
            ret = [r[0] if scalar_key in r.attrs else r[:] for r in ret]

            # handle multiple results
            ret = ret[0] if len(names) == 1 else tuple(ret)

            return ret

    def put(self, result, namespace, *args, **kwargs):
        """Store results in the cache."""

        # obtain group holding cached results for the given namespace
        namespace_grp = self.backend.require_group(namespace)

        # obtain a key for the given arguments
        key = self.key_smith.cut(*args, **kwargs)

        # obtain group holding cached results for the given key
        result_grp = namespace_grp.require_group(key)

        # reset storage
        if success_key in result_grp.attrs:
            del result_grp.attrs[success_key]
        for n in result_grp.keys():
            del result_grp[n]

        # handle multiple results
        if isinstance(result, tuple):
            if len(result) > 99:
                raise RuntimeError('too many results')
            names = ['f%02d' % i for i in range(len(result))]
            results = result
        else:
            names = ['f01']
            results = [result]

        # proceed with storing result in cache
        for n, r in zip(names, results):
            if np.isscalar(r):
                # handle scalars
                kwargs = self.dataset_kwargs.copy()
                kwargs['compressor'] = None
                ds = result_grp.create_dataset(n, data=[r], **kwargs)
                ds.attrs[scalar_key] = True
            else:
                ds = result_grp.create_dataset(n, data=r, **self.dataset_kwargs)
            if self.verbose:
                log('[%s] stored' % namespace, repr(ds))

        # mark caching operation as successfully completed
        result_grp.attrs[success_key] = True

    def delete(self, namespace, *args, **kwargs):
        """Delete an item from the cache."""

        try:

            # obtain group holding cached results for the given namespace
            namespace_grp = self.backend[namespace]

            # obtain a key for the given arguments
            key = self.key_smith.cut(*args, **kwargs)

            # delete the cached result
            del namespace_grp[key]

            if self.verbose:
                log('[%s] deleted cache entry' % namespace, args, kwargs)

        except KeyError:
            if self.verbose:
                log('[%s] cache miss' % namespace, args, kwargs)
            raise CacheMiss(namespace, *args, **kwargs)

    def clear(self, namespace=None):
        if namespace:
            if namespace in self.backend:
                del self.backend[namespace]
                if self.verbose:
                    log('[%s] cleared cache' % namespace)
        else:
            for k in sorted(self.backend.keys()):
                del self.backend[k]
                if self.verbose:
                    log('[%s] cleared cache' % k)

    def memoize(self, f):
        """Decorator to memoize a function."""

        namespace = f.__name__

        def wrapper(*args, **kwargs):
            try:
                result = self.get(namespace, *args, **kwargs)
            except CacheMiss:
                result = f(*args, **kwargs)
                self.put(result, namespace, *args, **kwargs)
            return result

        def _clear():
            self.clear(namespace)

        def _delete(*args, **kwargs):
            self.delete(namespace, *args, **kwargs)

        wrapper.clear = _clear
        wrapper.delete = _delete

        return wrapper
