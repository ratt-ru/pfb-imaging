from setuptools import setup, find_packages
import pfb

with open("README.rst", "r") as fh:
    long_description = fh.read()

requirements = [
                'numpy',
                'tornado==6.1',
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
                'bokeh==2.4.3',
                'graphviz',
                'nifty7',
                'sympy',
                "codex-africanus[complete]",
                "dask-ms[xarray, zarr, s3]",
                "finufft",
                "QuartiCal"
                "@git+https://github.com/ratt-ru/QuartiCal.git"
                "@v0.2.0-dev",

                "packratt"
                "@git+https://github.com/ratt-ru/packratt.git"
                "@master",

                "stimela"
                "@git+https://github.com/caracal-pipeline/stimela2"
                "@master",

                "smoove"
                "@git+https://github.com/landmanbester/smoove.git"
                "@test_ci"
            ]


# If scabha is not pre-installed, add to requirements
# This is just a transitionary hack: next release of stimela will include scabha,
# so when that happens, we just add a stimela>=2 dependency at the top, and don't bother with this
# try:
#     import scabha
# except ImportError:
#     requirements.append('scabha')


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
     python_requires='>=3.8',
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
