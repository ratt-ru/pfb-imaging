from dataclasses import dataclass
import os.path
import glob
from typing import *
from scabha import configuratt
from scabha.cargo import Parameter
from omegaconf.omegaconf import OmegaConf


schema = None

@dataclass
class _CabInputsOutputs(object):
    inputs: Dict[str, Parameter]
    outputs: Dict[str, Parameter]

# load schema files
if schema is None:

    # all *.yaml files under pfb.parser will be loaded automatically

    files = glob.glob(os.path.join(os.path.dirname(__file__), "*.yaml"))

    structured = OmegaConf.structured(_CabInputsOutputs)

    tmp = configuratt.load_nested(files,
                                  structured=structured,
                                  config_class="PfbCleanCabs")

    # this is required since upgrade of scabha to caching branch.
    # tmp is a tuple containing the config object as the first element
    # and a set containing locations of .yaml configs for pfb workers
    schema = OmegaConf.create(tmp)




