"""Frozen PARNET wrapper — the validated local load recipe (from parnet_interp_probe.py).

The 7M weights are a full pickled v0.3.0 NewRBPNet object, but the local source is v0.1.1
(forward signatures differ; projection is None). We resolve this by:
  1. registering a stub `parnet` package pointing at the real source dir so the pickle resolves
     its classes WITHOUT running the heavy __init__ (which pulls lightning / TF / pybigwig);
  2. stubbing pytorch_lightning (only used for gin configurables we never touch);
  3. bypassing forward() and calling submodules directly: stem -> body -> [projection?] -> head.
Head = NewAdditiveMix -> {target, control, total (T,L), mix_coeff (T), penalty_loss}.
"""
from __future__ import annotations
import sys
import types
import torch

from .. import config

_LOADED = False


def _install_stubs():
    global _LOADED
    if _LOADED:
        return
    pkgdir = str(config.PARNET_PKG)
    if "parnet" not in sys.modules:
        pkg = types.ModuleType("parnet")
        pkg.__path__ = [pkgdir]
        sys.modules["parnet"] = pkg
    if "pytorch_lightning" not in sys.modules:
        def _mk(n):
            return type(n, (), {"__init__": lambda self, *a, **k: None})
        pl = types.ModuleType("pytorch_lightning")
        cb = types.ModuleType("pytorch_lightning.callbacks")
        lg = types.ModuleType("pytorch_lightning.loggers")
        cb.EarlyStopping = _mk("EarlyStopping"); cb.LearningRateMonitor = _mk("LearningRateMonitor")
        lg.TensorBoardLogger = _mk("TensorBoardLogger"); lg.WandbLogger = _mk("WandbLogger")
        pl.callbacks = cb; pl.loggers = lg
        sys.modules.update({"pytorch_lightning": pl,
                            "pytorch_lightning.callbacks": cb,
                            "pytorch_lightning.loggers": lg})
    import parnet.models  # noqa: F401 — registers the real classes for the unpickler
    _LOADED = True


class ParnetModel:
    """Frozen PARNET. Call .full(onehot) for the {target,control,total,mix_coeff} dict,
    .body_feats(onehot) for the (B,512,L) representation (the demo's stem+body embedding)."""

    def __init__(self, module, syms, device):
        self.m = module
        self.syms = syms                       # list[(symbol, cell)] in task order
        self.idx = {f"{s}_{c}": i for i, (s, c) in enumerate(syms)}
        self.device = device

    def track_index(self, symbol, cell=None):
        if cell:
            return self.idx.get(f"{symbol}_{cell}")
        return next((i for i, (s, _c) in enumerate(self.syms) if s == symbol), None)

    @torch.no_grad()
    def full(self, onehot):
        """onehot: (B,4,L) float -> dict of softmaxed profiles + mix_coeff."""
        h = self.m.stem(onehot); h = self.m.body(h)
        if getattr(self.m, "projection", None) is not None:
            h = self.m.projection(h)
        out = self.m.head(h)
        return {k: (torch.softmax(v, dim=2) if k in ("target", "control", "total") else v)
                for k, v in out.items()}

    def run_raw(self, onehot):
        """No-softmax forward through submodules — for attribution / custom losses (keeps grad)."""
        h = self.m.stem(onehot); h = self.m.body(h)
        if getattr(self.m, "projection", None) is not None:
            h = self.m.projection(h)
        return self.m.head(h)

    @torch.no_grad()
    def body_feats(self, onehot):
        """(B,512,L) stem+body representation (the lab's embedding path)."""
        return self.m.body(self.m.stem(onehot))


def load_parnet(weights=None, device=None) -> ParnetModel:
    _install_stubs()
    weights = str(weights or config.PARNET_WEIGHTS)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    m = torch.load(weights, map_location=device, weights_only=False).to(torch.float32).eval()
    idx2sym = torch.load(str(config.PARNET_IDX2SYM), map_location="cpu", weights_only=False)
    syms = [tuple(idx2sym[i]) for i in range(len(idx2sym))]
    return ParnetModel(m, syms, device)
