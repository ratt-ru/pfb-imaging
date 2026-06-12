#!/bin/bash

hip-cargo generate-function --cab-file src/pfb_imaging/cabs/deconv.yml --config-file pyproject.toml --output-file src/pfb_imaging/cli/deconv.py
hip-cargo generate-function --cab-file src/pfb_imaging/cabs/degrid.yml --config-file pyproject.toml --output-file src/pfb_imaging/cli/degrid.py
hip-cargo generate-function --cab-file src/pfb_imaging/cabs/fluxtractor.yml --config-file pyproject.toml --output-file src/pfb_imaging/cli/fluxtractor.py
hip-cargo generate-function --cab-file src/pfb_imaging/cabs/grid.yml --config-file pyproject.toml --output-file src/pfb_imaging/cli/grid.py
hip-cargo generate-function --cab-file src/pfb_imaging/cabs/hci.yml --config-file pyproject.toml --output-file src/pfb_imaging/cli/hci.py
hip-cargo generate-function --cab-file src/pfb_imaging/cabs/init.yml --config-file pyproject.toml --output-file src/pfb_imaging/cli/init.py
hip-cargo generate-function --cab-file src/pfb_imaging/cabs/imager.yml --config-file pyproject.toml --output-file src/pfb_imaging/cli/imager.py
hip-cargo generate-function --cab-file src/pfb_imaging/cabs/kclean.yml --config-file pyproject.toml --output-file src/pfb_imaging/cli/kclean.py
hip-cargo generate-function --cab-file src/pfb_imaging/cabs/model2comps.yml --config-file pyproject.toml --output-file src/pfb_imaging/cli/model2comps.py
hip-cargo generate-function --cab-file src/pfb_imaging/cabs/restore.yml --config-file pyproject.toml --output-file src/pfb_imaging/cli/restore.py
hip-cargo generate-function --cab-file src/pfb_imaging/cabs/sara.yml --config-file pyproject.toml --output-file src/pfb_imaging/cli/sara.py
