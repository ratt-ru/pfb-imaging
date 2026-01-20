import glob
import os.path
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import *

from omegaconf.omegaconf import OmegaConf
from scabha import configuratt
from scabha.cargo import Parameter


def EmptyDictDefault():
    return field(default_factory=lambda: OrderedDict())


schema = None


@dataclass
class _CabInputsOutputs(object):
    # inputs: Dict[str, Parameter]
    # outputs: Dict[str, Parameter]
    inputs: Dict[str, Parameter] = EmptyDictDefault()
    outputs: Dict[str, Parameter] = EmptyDictDefault()
    policies: Optional[Dict[str, Any]] = None


# load schema files
if schema is None:
    # *.yaml files under pfb.parser will be loaded automatically
    # files that should not be included must have a .yml extension
    files = glob.glob(os.path.join(os.path.dirname(__file__), "*.yaml"))

    structured = OmegaConf.structured(_CabInputsOutputs)

    tmp = configuratt.load_nested(files, structured=structured, config_class="PfbCleanCabs", use_cache=False)

    # tmp is a tuple containing the config object as the first element
    # and a set containing locations of .yaml configs for pfb workers
    schema = OmegaConf.create(tmp[0])
