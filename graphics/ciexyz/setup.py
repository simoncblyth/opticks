# ../logit/setup.py for docs

def configuration(parent_package='', top_path=None):
    import numpy
    from numpy.distutils.misc_util import Configuration

    config = Configuration('ciexyz',
                           parent_package,
                           top_path)

    config.add_extension('ciexyz', ['ciexyz_numpy_ufunc.c'])

    return config

if __name__ == "__main__":
    from numpy.distutils.core import setup
    setup(configuration=configuration)


