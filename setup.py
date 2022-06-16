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

                "dask-ms[xarray, zarr]"
                "@git+https://github.com/ska-sa/dask-ms.git"
                "@master",

                "packratt"
                "@git+https://github.com/ratt-ru/packratt.git"
                "@master",

                "scabha"
                "@git+https://github.com/caracal-pipeline/scabha2"
                "@rich",

                "stimela"
                "@git+https://github.com/caracal-pipeline/stimela2.git"
                "@rich",


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
     python_requires='>=3.7',
     install_requires=requirements,
     classifiers=[
         "Programming Language :: Python :: 3",
         "License :: OSI Approved :: MIT License",
         "Operating System :: OS Independent",
     ],
     entry_points={'console_scripts':[
        'pfb = pfb.workers.main:cli',
        'pfbmisc = pfb.workers.experimental:cli'
        ]
     }


     ,
 )
