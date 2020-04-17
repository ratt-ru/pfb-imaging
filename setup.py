import setuptools
with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
     name='pfb',  
     version='0.0.1',
     scripts=[] ,
     author="Landman Bester",
     author_email="lbester@ska.ac.za",
     description="Pre-conditioned forward-backward CLEAN algorithm",
     long_description=long_description,
   long_description_content_type="text/markdown",
     url="https://github.com/ratt-ru/pfb-clean",
     packages=setuptools.find_packages(),
     install_requires=[
          'numpy',
          'scipy',
          'matplotlib',
          'numba',
          'astropy',
          'python-casacore',
          'dask',
          "dask[array]",
          "dask-ms[xarray]",
          'codex-africanus',
          'PyWavelets',
      ],
     classifiers=[
         "Programming Language :: Python :: 3",
         "License :: OSI Approved :: MIT License",
         "Operating System :: OS Independent",
     ],
 )