import setuptools
with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
     name='pfb',  
     version='0.0.1',
     scripts=['simple_spi_fitter.py', 'power_beam_maker.py', 'image_convolver.py', 'pfb.py', 'solve_x0.py', 'make_dirty.py', 'prep_data.py'] ,
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
          'codex-africanus >= 0.2.4',
          'PyWavelets',
          'ducc0 >= 0.4.0',
      ],
     classifiers=[
         "Programming Language :: Python :: 3",
         "License :: OSI Approved :: MIT License",
         "Operating System :: OS Independent",
     ],
 )