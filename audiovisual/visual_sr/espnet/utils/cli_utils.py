from distutils.util import strtobool as dist_strtobool


def strtobool(x):
    # distutils.util.strtobool returns integer, but it's confusing,
    return bool(dist_strtobool(x))
