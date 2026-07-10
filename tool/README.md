# HDF5 to GIF

Use `hdf5_to_gif.py` to inspect collected rollout videos from HDF5 files.

```bash
# Convert all demos in one HDF5 file.
python tool/hdf5_to_gif.py \
  --input /root/autodl-tmp/metaworld_m6_hdf5/button-press-v3.hdf5 \
  --output-dir /root/autodl-tmp/metaworld_gifs

# Convert one episode only.
python tool/hdf5_to_gif.py \
  --input /root/autodl-tmp/metaworld_m6_hdf5/button-press-v3.hdf5 \
  --episode demo_0

# Convert all HDF5 files in a directory.
python tool/hdf5_to_gif.py \
  --input /root/autodl-tmp/metaworld_m6_hdf5 \
  --output-dir /root/autodl-tmp/metaworld_gifs
```

Default image path inside each episode is `data/demo_x/image_primary`.
