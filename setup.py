from setuptools import setup, find_packages
import pfb

with open("README.rst", "r") as fh:
    long_description = fh.read()

requirements = [
                'matplotlib',
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

                "codex-africanus[complete]"
                "@git+https://github.com/landmanbester/codex-africanus.git"
                "@calutils",

                "dask-ms[xarray, zarr]"
                "@git+https://github.com/ska-sa/dask-ms.git"
                "@master",

                "packratt"
                "@git+https://github.com/ratt-ru/packratt.git"
                "@master",
            ]


setup(
     name='pfb-clean',
     version=pfb.__version__,
     author="Landman Bester",
     author_email="lbester@ska.ac.za",
     description="Pre-conditioned forward-backward CLEAN algorithm",
     long_description=long_description,
     long_description_content_type="text/markdown",
     url="https://github.com/ratt-ru/pfb-clean",
     packages=find_packages(),
     install_requires=requirements,
     classifiers=[
         "Programming Language :: Python :: 3",
         "License :: OSI Approved :: MIT License",
         "Operating System :: OS Independent",
     ],
     entry_points='''
                    [console_scripts]
                    pfb=pfb.workers.main:cli

     '''
     ,
 )
