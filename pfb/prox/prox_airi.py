"""
AIRI proximity operator
"""

from abc import ABC, abstractmethod
from typing import Any
import csv
import os
import torch
from onnx2torch import convert

# TODO: Add faceting functionality

class ProxOp(ABC):
    """
    Base class for proximity operator. Based on code from repo: github.com/basp-group/Small-scale-RI-imaging/ on 1 December 2025'

    This class provides the base functionality for a proximity operator.
    It defines common methods for various proximity operators, such as
    applying the operator to an input and updating the operator.
    The actual implementation of these methods are left to the subclasses.
    """

    def __init__(
        self,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float,
    ) -> None:
        """
        Initialize the ProxOp class.

        Args:
            device (torch.device, optional): The device for the tensors.
                Defaults to torch.device("cpu").
            dtype (torch.dtype, optional): The data type for the tensors.
                Defaults to torch.float.
        """
        self._dtype = dtype
        self._device = device

    @abstractmethod
    def __call__(self, x: torch.Tensor) -> Any:
        """
        Apply proximity operator to input.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            NotImplemented: This method should be implemented in a subclass.
        """
        return NotImplemented

    @abstractmethod
    def update(self, *args, **kwargs) -> Any:
        """
        Update proximity operator.

        This method should be implemented in a subclass.
        The input arguments can be arbitrary and should be specified in the subclass.
        """
        return NotImplemented

    def get_device(self) -> torch.device:
        """
        Return the device that the proximity operator is running on.

        Returns:
            torch.device: The device that the proximity operator is running on.
        """
        return self._device

    def get_data_type(self) -> torch.dtype:
        """
        Return the data type of the data that the proximity operator will accept and return.

        Returns:
            torch.dtype: The data type of the data that the proximity operator will accept
                and return.
        """
        return self._dtype


class ProxOpAIRI(ProxOp):
    """
    Largely based on code from repo: github.com/basp-group/Small-scale-RI-imaging/ on 1 December 2025'
    AIRI proximity operator

    This class implements the AIRI proximity operator which uses AIRI denoisers for regularisations.
    It uses a shelf of pre-trained AIRI denoisers and selects the appropriate one based on
    the estimated maximum intensitie of target image and the heuristic noise level.
    """

    def __init__(
        self,
        shelf_path: str,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float,
        verbose: bool = True,
    ) -> None:
        """
        Initializes the AIRI proximity operator with the given parameters.

        Args:
            shelf_path (str): Path to the CSV file containing the denoiser shelf.
            device (torch.device, optional): The device on which the computations are
                performed. Defaults to torch.device("cpu").
            dtype (torch.dtype, optional): The data type of the input. Defaults to torch.float.
            verbose (bool, optional): If True, print progress messages. Defaults to True.
        """
        super().__init__(device=device, dtype=dtype)

        self._net_scaling = 1.0
        self._shelf = {}
        self._network = None
        self._verbose = verbose
        # load paths of networks
        if not os.path.isfile(shelf_path):
            raise FileNotFoundError("Shelf file not found: " + shelf_path)
        with open(shelf_path, newline="", encoding="utf-8") as shelf_file:
            shelf_reader = csv.reader(shelf_file, delimiter=",")
            for row in shelf_reader:
                self._shelf[float(row[0])] = row[1]
                if not os.path.isfile(row[1]):
                    raise FileNotFoundError("Denoiser file not found: " + row[1])
        if not self._shelf:
            raise RuntimeError("Shelf is empty: " + shelf_path)
        self._shelf = dict(sorted(self._shelf.items()))

    @torch.no_grad()
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        result = self._network(x / self._net_scaling) * self._net_scaling
        return result

    def update(self, heuristic: float, peak_est: float) -> None:
        """
        Updates the denoiser selection and scaling factor based on the given
        heuristic noise level and maximum intensity.

        Args:
            heuristic (float): The heuristic noise level.
            peak_est (float): The estimated maximum intensity.
        """
        peak_min = 0
        peak_max = 0

        sigma_s = max(
            filter(lambda i: i <= heuristic / peak_est, self._shelf.keys()),
            default=None,
        )
        if sigma_s:
            peak_max = heuristic / sigma_s
            sigma_s1 = min(
                filter(lambda i: i > sigma_s, self._shelf.keys()), default=None
            )
            if sigma_s1:
                peak_min = heuristic / sigma_s1
        else:
            sigma_s = min(self._shelf.keys())
            peak_min = heuristic / sigma_s
            peak_max = float("inf")

        self._network = convert(self._shelf[sigma_s]).to(self._device).eval()
        self._net_scaling = heuristic / sigma_s

        if self._verbose:
            print(
                f"\nSHELF *** Inverse of the estimated target dynamic range: {heuristic/peak_est}",
                flush=True,
            )
            print(f"SHELF *** Using network: {self._shelf[sigma_s]}", flush=True)
            print(
                f"SHELF *** Peak value is expected in range: [{peak_min}, {peak_max}]",
                flush=True,
            )
            print(
                f"SHELF *** scaling factor applied to the image: {self._net_scaling}",
                flush=True,
            )

        return (peak_min, peak_max)


