import functools
import os
import pickle

cache_dict = {}


def cache_func(cache_file_path, ignore_load=False, ignore_save=False, verbose=True, is_instance_method=False):
    """
    Decorator which will use "cachefile" for caching the results of the decorated function "fn".
    The ignore options are used to ignore the load and save cache.
    supports pickle caching only.
    :param obj_id: the object id of the function's class. Use
    :param cache_file_path: file path to save the cache to
    :param ignore_load: (Default: False) Ignore existing cache, invoke function anyways.
    :param ignore_save: (Default: False) Ignore saving cache.
    :param verbose: print some messages regarding load and saving the cache.
    :return: a cache decorator
    """
    def function_to_name(fn):
        return f"{fn.__module__}.{fn.__qualname__}"

    def get_key(fn):
        return function_to_name(fn)

    def cache_decorator(fn):  # define a decorator for a function "fn"
        if fn not in cache_dict:
            cache_dict[get_key(fn)] = cache_file_path

        @functools.wraps(fn)
        def cache_wrapped(*args, **kwargs):  # define a wrapper that will finally call "fn" with all arguments
            # if cache exists -> load it and return its content
            cache_file = cache_dict[get_key(fn)]
            if not ignore_load and os.path.exists(cache_file):
                with open(cache_file, 'rb') as cachehandle:
                    if verbose:
                        print("Using cached result from '{}'".format(cache_file))
                    return pickle.load(cachehandle)
            # execute the function with all arguments passed
            res = fn(*args, **kwargs)
            # write to cache file
            if not ignore_save:
                with open(cache_file, 'wb') as cachehandle:
                    if verbose:
                        print("Saving result to cache '{}'".format(cache_file))
                    pickle.dump(res, cachehandle)
            return res

        return cache_wrapped

    return cache_decorator  # return this "customized" decorator that uses "cachefile"
