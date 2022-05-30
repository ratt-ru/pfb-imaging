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
                "@becd3572f86a0ad78b55540f25fce6e129976a29",

                "packratt"
                "@git+https://github.com/ratt-ru/packratt.git"
                "@aae9c681f647b8b9ad6d884258b7c4eceff58edd",

                "scabha"
                "@git+https://github.com/caracal-pipeline/scabha2"
                "@af3225e66161b523623c56f02c9d2570721c3d3c",

                "stimela"
                "@git+https://github.com/caracal-pipeline/stimela2.git"
                "@14c704d4f536814dee02dd80185f4870adec9de1"

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
