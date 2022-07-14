from setuptools import setup, find_packages
import pfb

with open("README.rst", "r") as fh:
    long_description = fh.read()

requirements = [
                'matplotlib',
                'ipython',
                'scikit-image',
                'dask[distributed]',
                'PyWavelets',
                'katbeam',
                'pytest >= 6.2.2',
                'numexpr',
                'pyscilog >= 0.1.2',
                'Click',
                'omegaconf',
                'bokeh',
                'graphviz',
                'nifty7',
                'sympy',
                'rich',
                "codex-africanus[complete]",

                "dask-ms[xarray, zarr, s3]"
                "@git+https://github.com/ska-sa/dask-ms.git"
                "@master",

                "packratt"
                "@git+https://github.com/ratt-ru/packratt.git"
                "@master",

            ]


# If scabha is not pre-installed, add to requirements
# This is just a transitionary hack: next release of stimela will include scabha,
# so when that happens, we just add a stimela>=2 dependency at the top, and don't bother with this
try:
    import scabha
except ImportError:
    requirements.append('scabha')


setup(
     name='pfb-clean',
     version=pfb.__version__,
     author="Landman Bester",
     author_email="lbester@sarao.ac.za",
     description="Pre-conditioned forward-backward CLEAN algorithm",
     long_description=long_description,
     long_description_content_type="text/markdown",
     url="https://github.com/ratt-ru/pfb-clean",
     packages=find_packages(),
     include_package_data=True,
     zip_safe=False,
     python_requires='>=3.7',
     install_requires=requirements,
     classifiers=[
         "Programming Language :: Python :: 3",
         "License :: OSI Approved :: MIT License",
         "Operating System :: POSIX :: Linux",
         "Topic :: Scientific/Engineering :: Astronomy",
     ],
     entry_points={'console_scripts':[
        'pfb = pfb.workers.main:cli',
        'pfbmisc = pfb.workers.experimental:cli'
        ]
     }


     ,
 )
