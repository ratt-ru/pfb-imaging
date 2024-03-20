pfb-clean
=========
Preconditioned forward-backward clean algorithm.

Install the package by cloning and running

:code:`$ pip install -e pfb-clean/`

Note casacore needs to be installed on the system for this to work.

You will probably may also need to update pip eg.

:code:`$ pip install -U pip setuptools wheel`

For maximum performance it is strongly recommended to install ducc in
no binary mode eg

:code:`$ git clone https://gitlab.mpcdf.mpg.de/mtr/ducc.git`
:code:`$ pip install -e ducc`

You may also have to make numba aware of the tbb layer by doing

:code:`$ pip install tbb`
:code:`$ export LD_LIBRARY_PATH=/path/to/venv/lib`

see eg. https://github.com/ratt-ru/QuartiCal/issues/268

If you find any of this useful please cite (for now)

https://arxiv.org/abs/2101.08072
