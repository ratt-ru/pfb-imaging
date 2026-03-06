Example augmenting recipes
~~~~~~~~~~~~~~~~~~~~~~~~~~
This folder contains config files that augment the imaging recipes contained in the `recipes` module.
The `recipes` module contains generic API style recipes with some sensible defaults.
These can be augmented with config files similar to the ones contained in this folder to:

* Set project level parameters to significantly shorten your terminal command.
* Overwrite defaults in the API recipe.
* Set any additonal options (e.g. backend options) that have intentially been left unset in the API recipe.

These config files can be used as examples for your own fields.
Please consider submitting your recipe for inclusion if you've managed to successfully use `pfb-imaging` for your own projects.
Some attempt will be mode to keep these recipes up to date as parameters change in the underlying API.
Recipes, like cabs, link to the default container associated with the package and therefore should be fully reproducible with a specific version of `pfb-imaging`.

ESO137
------


Galactic Center
---------------
The [SGRA recipe](SGRA.yml) has preconfigured config for three Galactic center pointings at UHF band. The recipe expects the data to live in a sub-directory called `msdir` (data available on request). Once `pfb-imaging` and `stimela` are installed, and the data has beem placed in the `msdir` sub-directory, the recipe can be invoked using e.g.

```bash
stimela run pfb_imaging.recipes::sara.yml SGRA.yml gosara obs=<obs>
```
where `<obs>` can be one of `sgra`, `gcx17` or `gcx30`. Outputs will be placed in folders called `output/<obs>` by default, with fits files and logs appearing in `output/<obs>/fits` and `output/<obs>/logs`, respectively.

Tarantula (A.K.A. 30 Doradus)
-----------------------------
