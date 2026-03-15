"""Step 1: Discover all available ESRF fragment volumes on S3.

Catalogs every zarr volume: resolution, propagation distance, energy, array shape.
"""
import s3fs
import zarr
import json

fs = s3fs.S3FileSystem(anon=True)
bucket = 'vesuvius-challenge-open-data'

targets = ['PHerc0500P2', 'PHerc0343P', 'PHerc0009B']

all_volumes = []

for frag in targets:
    vol_path = f'{bucket}/{frag}/volumes'
    items = fs.ls(vol_path)

    for item in sorted(items):
        name = item.split('/')[-1]
        if not name.endswith('.zarr'):
            continue

        # Parse name: timestamp-pixelsize-propdist-energy-masked.zarr
        parts = name.replace('.zarr', '').split('-')
        # Format: 20250526151718-2.215um-0.4m-111keV-masked
        timestamp = parts[0]
        pixel_size = parts[1]
        prop_dist = parts[2]
        energy = parts[3]
        masked = parts[4] if len(parts) > 4 else ''

        vol_info = {
            'fragment': frag,
            'name': name,
            'pixel_size': pixel_size,
            'prop_distance': prop_dist,
            'energy': energy,
            's3_path': item,
        }

        # Try to read zarr metadata for shape/dtype
        store = s3fs.S3Map(root=item, s3=fs, check=False)
        try:
            root = zarr.open(store, mode='r')
            # OME-Zarr has multiscale arrays at /0, /1, etc.
            if hasattr(root, 'attrs'):
                multiscales = root.attrs.get('multiscales', [])
                if multiscales:
                    datasets = multiscales[0].get('datasets', [])
                    vol_info['n_scales'] = len(datasets)

            # Read the highest resolution array
            if '0' in root:
                arr = root['0']
                vol_info['shape'] = arr.shape
                vol_info['dtype'] = str(arr.dtype)
                vol_info['chunks'] = arr.chunks
            elif isinstance(root, zarr.Array):
                vol_info['shape'] = root.shape
                vol_info['dtype'] = str(root.dtype)
                vol_info['chunks'] = root.chunks
        except Exception as e:
            vol_info['error'] = str(e)

        all_volumes.append(vol_info)

# Print summary
print("=" * 80)
print("ESRF FRAGMENT VOLUME CATALOG")
print("=" * 80)

for frag in targets:
    vols = [v for v in all_volumes if v['fragment'] == frag]
    print(f"\n{'─'*80}")
    print(f"  {frag}  ({len(vols)} volumes)")
    print(f"{'─'*80}")
    for v in vols:
        print(f"\n  Volume: {v['name']}")
        print(f"    Pixel size:    {v['pixel_size']}")
        print(f"    Prop distance: {v['prop_distance']}")
        print(f"    Energy:        {v['energy']}")
        if 'shape' in v:
            print(f"    Shape:         {v['shape']}")
            print(f"    Dtype:         {v['dtype']}")
            print(f"    Chunks:        {v['chunks']}")
        if 'n_scales' in v:
            print(f"    Scales:        {v['n_scales']}")
        if 'error' in v:
            print(f"    ERROR:         {v['error']}")

# Summary table
print(f"\n\n{'='*80}")
print("SUMMARY TABLE")
print(f"{'='*80}")
print(f"{'Fragment':<14} {'Pixel Size':<12} {'Prop Dist':<10} {'Energy':<8} {'Shape'}")
print(f"{'─'*14} {'─'*12} {'─'*10} {'─'*8} {'─'*30}")
for v in all_volumes:
    shape_str = str(v.get('shape', 'N/A'))
    print(f"{v['fragment']:<14} {v['pixel_size']:<12} {v['prop_distance']:<10} {v['energy']:<8} {shape_str}")
