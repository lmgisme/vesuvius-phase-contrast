"""OME-Zarr streaming loader for ESRF fragment volumes.

Loads individual 2D slices (or small ROIs) directly from S3
without downloading the full volume. Uses zarr + s3fs for
lazy access with 128^3 chunking.

Usage:
    from src.io_zarr import ESRFVolume, list_volumes

    # List all available volumes
    vols = list_volumes()

    # Open a volume (no data downloaded yet)
    vol = ESRFVolume('PHerc0500P2', '9.362um-1.2m-113keV')

    # Read a single axial slice (z=3500) at full resolution
    slice_data = vol.get_slice(3500, axis=0, scale=0)

    # Read a small ROI
    roi = vol.get_roi(z=3500, y=slice(2000, 3000), x=slice(2000, 3000), scale=0)

    # Read at a lower resolution (scale=3 is 8x downsampled)
    thumb = vol.get_slice(200, axis=0, scale=3)
"""
import s3fs
import zarr
import numpy as np

BUCKET = 'vesuvius-challenge-open-data'

# Complete catalog of ESRF fragment volumes
VOLUME_CATALOG = {
    'PHerc0500P2': {
        '2.215um-0.4m-111keV': '20250526151718-2.215um-0.4m-111keV-masked.zarr',
        '4.317um-1.2m-111keV': '20250528085330-4.317um-1.2m-111keV-masked.zarr',
        '9.362um-1.2m-113keV': '20250820143440-9.362um-1.2m-113keV-masked.zarr',
    },
    'PHerc0343P': {
        '8.640um-1.2m-116keV': '20250521134555-8.640um-1.2m-116keV-masked.zarr',
        '2.215um-0.4m-111keV': '20260304131111-2.215um-0.4m-111keV-masked.zarr',
    },
    'PHerc0009B': {
        '8.640um-1.2m-116keV': '20250521125136-8.640um-1.2m-116keV-masked.zarr',
        '2.401um-0.3m-77keV': '20250820154339-2.401um-0.3m-77keV-masked.zarr',
    },
}

# Full-resolution shapes (z, y, x) for reference
VOLUME_SHAPES = {
    'PHerc0500P2/2.215um-0.4m-111keV': (28096, 18209, 18209),
    'PHerc0500P2/4.317um-1.2m-111keV': (15838, 9423, 9423),
    'PHerc0500P2/9.362um-1.2m-113keV': (7057, 4196, 4196),
    'PHerc0343P/8.640um-1.2m-116keV': (5398, 5057, 5057),
    'PHerc0343P/2.215um-0.4m-111keV': (20714, 13155, 13155),
    'PHerc0009B/8.640um-1.2m-116keV': (9598, 7837, 7837),
    'PHerc0009B/2.401um-0.3m-77keV': (29112, 28259, 28259),
}


def list_volumes():
    """Print all available volumes."""
    for frag, vols in VOLUME_CATALOG.items():
        print(f"\n{frag}:")
        for key, zarr_name in vols.items():
            shape = VOLUME_SHAPES.get(f'{frag}/{key}', 'unknown')
            print(f"  {key:30s}  shape={shape}")


class ESRFVolume:
    """Streaming access to a single ESRF OME-Zarr volume on S3."""

    def __init__(self, fragment: str, volume_key: str):
        """
        Args:
            fragment: e.g. 'PHerc0500P2'
            volume_key: e.g. '9.362um-1.2m-113keV'
        """
        if fragment not in VOLUME_CATALOG:
            raise ValueError(f"Unknown fragment: {fragment}. "
                             f"Available: {list(VOLUME_CATALOG.keys())}")
        if volume_key not in VOLUME_CATALOG[fragment]:
            raise ValueError(f"Unknown volume key: {volume_key}. "
                             f"Available: {list(VOLUME_CATALOG[fragment].keys())}")

        self.fragment = fragment
        self.volume_key = volume_key
        self.zarr_name = VOLUME_CATALOG[fragment][volume_key]
        self.s3_path = f"{BUCKET}/{fragment}/volumes/{self.zarr_name}"

        # Parse scan parameters from the key
        parts = volume_key.split('-')
        self.pixel_size_um = float(parts[0].replace('um', ''))
        self.prop_distance_m = float(parts[1].replace('m', ''))
        self.energy_keV = float(parts[2].replace('keV', ''))

        # Connect to S3
        self._fs = s3fs.S3FileSystem(anon=True)
        self._store = s3fs.S3Map(root=self.s3_path, s3=self._fs, check=False)
        self._root = zarr.open(self._store, mode='r')

        # Get multiscale info
        self._arrays = {}
        multiscales = self._root.attrs.get('multiscales', [])
        if multiscales:
            datasets = multiscales[0].get('datasets', [])
            self.n_scales = len(datasets)
            for ds in datasets:
                path = ds['path']
                self._arrays[int(path)] = self._root[path]
        else:
            # Fallback: try numeric keys
            self.n_scales = 0
            for key in self._root.keys():
                try:
                    self._arrays[int(key)] = self._root[key]
                    self.n_scales += 1
                except ValueError:
                    pass

        self.shape = self._arrays[0].shape
        self.dtype = self._arrays[0].dtype
        self.chunks = self._arrays[0].chunks

        print(f"Opened: {fragment}/{self.zarr_name}")
        print(f"  Shape: {self.shape}, dtype: {self.dtype}, chunks: {self.chunks}")
        print(f"  Scales: {self.n_scales} (0=full, {self.n_scales-1}=coarsest)")
        print(f"  Pixel: {self.pixel_size_um} µm, Prop: {self.prop_distance_m} m, "
              f"Energy: {self.energy_keV} keV")

    def get_array(self, scale: int = 0):
        """Get the zarr array at a given scale level (0=full resolution)."""
        if scale not in self._arrays:
            raise ValueError(f"Scale {scale} not available. Have: {list(self._arrays.keys())}")
        return self._arrays[scale]

    def get_slice(self, index: int, axis: int = 0, scale: int = 0) -> np.ndarray:
        """Read a single 2D slice from the volume.

        Args:
            index: Slice index along the given axis.
            axis: 0=axial (z), 1=coronal (y), 2=sagittal (x).
            scale: Resolution level. 0=full, higher=downsampled.

        Returns:
            2D numpy array (uint8).
        """
        arr = self.get_array(scale)
        if axis == 0:
            return np.array(arr[index, :, :])
        elif axis == 1:
            return np.array(arr[:, index, :])
        elif axis == 2:
            return np.array(arr[:, :, index])
        else:
            raise ValueError(f"axis must be 0, 1, or 2, got {axis}")

    def get_roi(self, z=None, y=None, x=None, scale: int = 0) -> np.ndarray:
        """Read a sub-region of the volume.

        Args:
            z, y, x: int (single index) or slice objects. None = all.
            scale: Resolution level.

        Returns:
            numpy array (2D or 3D depending on indexing).
        """
        arr = self.get_array(scale)
        z = z if z is not None else slice(None)
        y = y if y is not None else slice(None)
        x = x if x is not None else slice(None)
        return np.array(arr[z, y, x])

    def scale_shape(self, scale: int) -> tuple:
        """Get the shape at a given scale level."""
        return self.get_array(scale).shape

    def __repr__(self):
        return (f"ESRFVolume({self.fragment}/{self.volume_key}, "
                f"shape={self.shape}, scales={self.n_scales})")
