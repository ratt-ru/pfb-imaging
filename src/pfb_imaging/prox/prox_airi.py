"""
AIRI proximity operator
"""

import csv
import os

import torch
from onnx2torch import convert

# TODO: Add faceting functionality


class ProxOpAIRI:
    """
    AIRI proximity operator.

    This class implements the AIRI proximity operator which uses AIRI denoisers for regularization.
    It uses a shelf of pre-trained AIRI denoisers and selects the appropriate one based on
    the estimated maximum intensity of target image and the heuristic noise level.

    Based on code from github.com/basp-group/Small-scale-RI-imaging
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
                The CSV should have format: sigma,/path/to/network.onnx
            device (torch.device, optional): The device on which the computations are
                performed. Defaults to torch.device("cpu").
            dtype (torch.dtype, optional): The data type of the input. Defaults to torch.float.
            verbose (bool, optional): If True, print progress messages. Defaults to True.
        """
        self._dtype = dtype
        self._device = device
        self._net_scaling = 1.0
        self._shelf = {}
        self._network = None
        self._verbose = verbose

        # Load denoiser shelf from CSV
        if not os.path.isfile(shelf_path):
            raise FileNotFoundError(f"Shelf file not found: {shelf_path}")

        with open(shelf_path, newline="", encoding="utf-8") as shelf_file:
            shelf_reader = csv.reader(shelf_file, delimiter=",")
            for row in shelf_reader:
                if len(row) < 2:
                    continue  # Skip empty or malformed rows
                sigma = float(row[0])
                network_path = row[1].strip()
                self._shelf[sigma] = network_path
                if not os.path.isfile(network_path):
                    raise FileNotFoundError(f"Denoiser file not found: {network_path}")

        if not self._shelf:
            raise RuntimeError(f"Shelf is empty: {shelf_path}")

        # Sort shelf by sigma values for efficient lookup
        self._shelf = dict(sorted(self._shelf.items()))

        if self._verbose:
            print(f"Loaded {len(self._shelf)} denoisers from shelf: {shelf_path}")
            print(f"Sigma values: {list(self._shelf.keys())}")

    @torch.no_grad()
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the AIRI denoiser to the input tensor.

        Args:
            x (torch.Tensor): The input tensor to denoise.

        Returns:
            torch.Tensor: The denoised tensor.

        Raises:
            RuntimeError: If update() has not been called to initialize the network.
        """
        if self._network is None:
            raise RuntimeError("Network not initialized. Call update() before applying the prox operator.")

        # Apply network with scaling: denoise(x/scale) * scale
        result = self._network(x / self._net_scaling) * self._net_scaling
        return result

    def update(self, heuristic: float, peak_est: float) -> tuple[float, float]:
        """
        Update the denoiser selection and scaling factor based on the given
        heuristic noise level and estimated peak intensity.

        The method selects the denoiser whose noise level (sigma) is closest to
        but not exceeding heuristic/peak_est, then sets the scaling factor to
        normalize the image intensity to match the denoiser's training scale.

        Args:
            heuristic (float): The heuristic noise level.
            peak_est (float): The estimated maximum intensity of the image.

        Returns:
            tuple[float, float]: (peak_min, peak_max) - the valid range of peak values
                for the selected denoiser.
        """
        peak_min = 0.0
        peak_max = 0.0

        # Find largest sigma_s such that sigma_s <= heuristic / peak_est
        sigma_s = max(
            filter(lambda i: i <= heuristic / peak_est, self._shelf.keys()),
            default=None,
        )

        if sigma_s:
            # Found a denoiser within the dynamic range
            peak_max = heuristic / sigma_s
            # Find the next higher sigma to determine lower bound on peak
            sigma_s1 = min(filter(lambda i: i > sigma_s, self._shelf.keys()), default=None)
            if sigma_s1:
                peak_min = heuristic / sigma_s1
        else:
            # No denoiser found, use the lowest sigma (most aggressive denoising)
            sigma_s = min(self._shelf.keys())
            peak_min = heuristic / sigma_s
            peak_max = float("inf")

        # Load and initialize the selected network
        self._network = convert(self._shelf[sigma_s]).to(self._device).eval()
        self._net_scaling = heuristic / sigma_s

        if self._verbose:
            print(
                f"\nSHELF *** Inverse of the estimated target dynamic range: {heuristic / peak_est:.6e}",
                flush=True,
            )
            print(f"SHELF *** Using network: {self._shelf[sigma_s]}", flush=True)
            print(
                f"SHELF *** Peak value is expected in range: [{peak_min:.6e}, {peak_max:.6e}]",
                flush=True,
            )
            print(
                f"SHELF *** Scaling factor applied to the image: {self._net_scaling:.6e}",
                flush=True,
            )

        return (peak_min, peak_max)

    @torch.no_grad()
    def call_tiled(
        self,
        x: torch.Tensor,
        tile_size: int,
        margin: int,
    ) -> torch.Tensor:
        """
        Apply the AIRI denoiser to a large image using overlapping tiles.

        Splits the input into overlapping tiles, processes each through the
        denoiser, then recombines them. The margin provides overlap so that
        edge artifacts are avoided when tiles are merged.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, channel, npix, npix).
            tile_size (int): Size of the inner region of each tile (the part kept).
            margin (int): Overlap on each side. Full tile size is (tile_size + 2*margin).

        Returns:
            torch.Tensor: Denoised tensor with same shape as input.
        """
        if self._network is None:
            raise RuntimeError("Network not initialized. Call update() before applying the prox operator.")

        npix = x.shape[-1]
        full_tile_size = tile_size + 2 * margin

        # Calculate number of tiles and padding needed
        n_tiles = int((npix + tile_size - 1) // tile_size)
        padded_size = n_tiles * tile_size + 2 * margin
        pad_needed = padded_size - npix

        # Pad input with zeros
        x_padded = torch.nn.functional.pad(
            x, (margin, pad_needed - margin, margin, pad_needed - margin), mode="constant", value=0
        )

        # Build output on regular grid, then crop
        grid_size = n_tiles * tile_size
        output_grid = torch.zeros((*x.shape[:-2], grid_size, grid_size), dtype=x.dtype, device=x.device)

        # Process tiles
        for i in range(n_tiles):
            for j in range(n_tiles):
                h_start = i * tile_size
                w_start = j * tile_size
                tile = x_padded[..., h_start : h_start + full_tile_size, w_start : w_start + full_tile_size]

                # Apply denoiser with scaling
                tile_denoised = self._network(tile / self._net_scaling) * self._net_scaling

                # Trim margins and place in output grid
                inner = tile_denoised[..., margin : margin + tile_size, margin : margin + tile_size]
                output_grid[..., h_start : h_start + tile_size, w_start : w_start + tile_size] = inner

        # Crop to original size
        return output_grid[..., :npix, :npix]

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
