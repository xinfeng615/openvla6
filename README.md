# openvla-zero

## MetaWorld M6 workflow

The M6 suite uses these detailed task instructions:

- `peg-insert-side-v3`: `insert a peg sideways`
- `basketball-v3`: `dunk the basketball into the basket`
- `coffee-pull-v3`: `pull a mug from a coffee machine`
- `pick-place-wall-v3`: `pick a puck, bypass a wall, and place the puck`
- `pick-out-of-hole-v3`: `pick up a puck from a hole`
- `box-close-v3`: `grasp the cover and close the box with it`

Main commands:

```bash
python collect_metaworld_m6_data.py
python metaworld_m6_50e_dataset_builder.py
bash finetune_m6.sh
bash eval_m6.sh
bash eval_m6_zeroshot.sh
```

The RLDS dataset name is `metaworld_m6_50e`. W&B projects default to
`m6-finetune` and `m6-eval`.
