# cython: language_level=3, boundscheck=False

import subprocess
import os
from distutils.extension import Extension
from setuptools import setup

import numpy as np
from Cython.Build import cythonize


def write_version_file(version):
    """ Writes a file with version information to be used at run time

    Parameters
    ----------
    version: str
        A string containing the current version information

    Returns
    -------
    version_file: str
        A path to the version file

    """
    try:
        git_log = subprocess.check_output(
            ['git', 'log', '-1', '--pretty=%h %ai']).decode('utf-8')
        git_diff = (subprocess.check_output(['git', 'diff', '.']) +
                    subprocess.check_output(
                        ['git', 'diff', '--cached', '.'])).decode('utf-8')
        if git_diff == '':
            git_status = '(CLEAN) ' + git_log
        else:
            git_status = '(UNCLEAN) ' + git_log
    except Exception as e:
        print("Unable to obtain git version information, exception: {}"
              .format(e))
        git_status = 'release'

    version_file = '.version'
    if not os.path.isfile(version_file):
        with open(f"coarse_psd_matrix/{version_file}", 'w+') as f:
            f.write('{}: {}'.format(version, git_status))

    return version_file


def get_long_description():
    """ Finds the README and reads in the description """
    here = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(here, 'README.md')) as f:
        long_description = f.read()
    return long_description


# get version info from __init__.py
def readfile(filename):
    with open(filename) as fp:
        filecontents = fp.read()
    return filecontents


VERSION = '0.1.0'
version_file = write_version_file(VERSION)
long_description = get_long_description()

setup(
    name="coarse_psd_matrix",
    description='Code to compute coarsened PSD matrices',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/ColmTalbot/psd-covariance-matrices',
    author='Colm Talbot',
    author_email='talbotcolm@gmail.com',
    license="MIT",
    version=VERSION,
    packages=["coarse_psd_matrix"],
    package_dir={'coarse_psd_matrix': 'coarse_psd_matrix'},
    package_data={'coarse_psd_matrix': ['coarse_gpu.cu', version_file]},
    ext_modules=cythonize([
        Extension("coarse_psd_matrix.coarse_cpu", ["coarse_psd_matrix/coarse_cpu.pyx"], include_dirs=[np.get_include()])
    ]),
    python_requires='>=3.6',
    install_requires=["numpy", "cython", "gwpopulation", "scipy<=1.7"],
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent"
    ]
)
