from setuptools import setup, find_packages
import pfb

with open("README.rst", "r") as fh:
    long_description = fh.read()

requirements = [
                'numpy <= 2.0.0',
                'scikit-image',
                'PyWavelets',
                'katbeam',
                'numexpr',
                'pyscilog >= 0.1.2',
                'Click',
                "ducc0"
                "@git+https://github.com/mreineck/ducc.git"
                "@tweak_wgridder_conventions",
                "QuartiCal"
                "@git+https://github.com/ratt-ru/QuartiCal.git"
                "@unpinneddeps",
                "sympy",
                "stimela >= 2.0rc18",
                "streamjoy >= 0.0.8",
                "codex-africanus[complete] >= 0.3.7",
                "tbb",
                "jax[cpu]",
                "ipycytoscape",
                "lz4",
                "ipdb",
                "psutil"
            ]


setup(
     name='pfb-imaging',
     version=pfb.__version__,
     author="Landman Bester",
     author_email="lbester@sarao.ac.za",
     description="Radio interferometric imaging suite base on the pre-conditioned forward-backward algorithm",
     long_description=long_description,
     long_description_content_type="text/markdown",
     url="https://github.com/ratt-ru/pfb-imaging",
     packages=find_packages(),
     include_package_data=True,
     zip_safe=False,
     python_requires='>=3.9',
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
