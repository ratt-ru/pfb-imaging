from setuptools import setup, find_packages
import pfb

with open("README.rst", "r") as fh:
    long_description = fh.read()

requirements = [
                'numpy',
                'matplotlib',
                'ipython',
                'scikit-image',
                'dask[distributed,diagnostics]',
                'PyWavelets',
                'katbeam',
                'pytest >= 6.2.2',
                'numexpr',
                'pyscilog >= 0.1.2',
                'Click',
                'omegaconf',
                "codex-africanus[dask, scipy, astropy, python-casacore, ducc0]",
                "dask-ms[xarray, zarr, s3]",
                # "stimela==2.0rc6",
                # "QuartiCal",
                "sympy",
                # "bokeh  >= 2.4.2, < 3",
                # "bokeh  >= 3.1.1",
                "ipdb",

                "QuartiCal"
                "@git+https://github.com/ratt-ru/QuartiCal.git"
                "@v0.2.1-degridder"
                # "stimela"
                # "@git+https://github.com/caracal-pipeline/stimela.git"
                # "@FIASCO3"

            ]


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
        'pfb = pfb.workers.main:cli'
        ]
     }


     ,
 )
