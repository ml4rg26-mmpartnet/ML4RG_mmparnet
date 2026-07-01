# `mmpartnet.data` — swappable data layer (team sketch)

One interface, many backends. An experiment asks for `{rbp, window, sequence, target}` records and does
not care where they came from. Swap the SOURCE by name; swap the target FORMAT by `cfg.target`. Populate
only what your experiment needs; leave the rest as stubs (they carry fill instructions).

```python
from mmpartnet.data import DataConfig, get_source, iter_records
cfg = DataConfig(source="encode_bigwig", group="QKI,PTBP1", cell="HepG2", nwin=8, target="density")
for rec in iter_records(get_source(cfg.source, cfg)):
    rec["sequence"]  # str, len lwin       -> one-hot -> model
    rec["target"]    # np.ndarray, len lwin -> a probability profile over the window
```

## Pieces

| file | role |
|------|------|
| `base.py` | `Window`, `DataConfig`, `DataSource` ABC (4 methods: `rbps`, `windows`, `sequence`, `observed`) |
| `registry.py` | name -> source class; `get_source(name, cfg)`, `list_sources()` |
| `preprocess.py` | raw observed signal -> probability profile, keyed by `cfg.target` (`density`/`counts`/`hfds`) |
| `loader.py` | `iter_records(source)`: applies preprocessing + the `>= min_sum` read filter, caps at `nwin`/RBP |
| `sources/encode_bigwig.py` | POPULATED demo source (public ENCODE, read remotely) |
| `sources/{encode_bam_counts,hfds,local_pt}.py` | STUBS with fill recipes |
| `multimodal.py` | flattened RNA-window/RBP-cell dataset and collator for FiLM and cross-attention experiments |

## Multimodal branch note

The PARNET HFDS stores one RNA window with labels for all RBP-cell tracks. The
conditional multimodal experiments train on a flattened view:

```text
PARNET window i + RBP-cell track j -> one training example
```

`multimodal.py` provides this view through `ParnetMultimodalDataset` and
`MultimodalCollator`. It handles:

- RNA sequence one-hot encoding and valid-position masks
- extraction of one eCLIP/control track from sparse PARNET labels
- PureCLIP/narrowPeak binary labels when supplied
- RBP/cell metadata and stable cell indices
- ProtT5 H5 lookup through the track-to-protein map
- pooled protein vectors for FiLM-style heads
- padded residue-level protein tensors and protein masks for cross-attention

## Add a source

1. Drop `sources/my_source.py`, subclass `DataSource`, implement the 4 methods, decorate with
   `@register("my_source")`.
2. Import it in `sources/__init__.py` so `@register` fires.
3. `get_source("my_source", cfg)` now works; nothing downstream changes.

## Add a target format

Register one function in `preprocess.py`:

```python
@register("myfmt")
def _myfmt(obs):
    ...  # raw per-nt signal -> non-negative profile summing to 1
```

Then set `cfg.target="myfmt"`. See `../../../docs/DATA_INVENTORY.md` for which source is the established
version and which is a surrogate (and why each surrogate is a valid sanity check).
