from setuptools import setup, find_packages
import pfb

with open("README.rst", "r") as fh:
    long_description = fh.read()

requirements = [
                'matplotlib',
                'scikit-image',
                'codex-africanus[complete] >= 0.2.10',
                'dask-ms[xarray] >= 0.2.6',
                'PyWavelets',
                'zarr',
                'katbeam',
                'packratt >= 0.1.3',
                'pytest >= 6.2.2',
                'numexpr',
                'pyscilog >= 0.1.2',
                'Click',
                'omegaconf',
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
                    pfbclean=pfb.pfbclean:main

     '''
     ,
 )