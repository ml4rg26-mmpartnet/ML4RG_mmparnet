"""Source backends. Importing this package runs each module so its `@register(name)` fires and the
source becomes reachable via `get_source(name, cfg)`. Add a backend = drop a module here + import it.

  encode_bigwig    POPULATED  public ENCODE RPM read-density bigWig (the demo path, runs now)
  encode_bam_counts  STUB     established single-nt 5' crosslink counts from ENCODE BAMs (lab-gated target)
  hfds               STUB     lab encode.filtered / HuggingFace dataset (canonical PARNET substrate)
  local_pt           STUB     pre-tiled ParnetDataElement .pt shards (offline / CI / teammate handoff)
"""
from __future__ import annotations

from . import encode_bigwig      # noqa: F401  (populated demo source)
from . import encode_bam_counts  # noqa: F401  (stub)
from . import hfds               # noqa: F401  (stub)
from . import local_pt           # noqa: F401  (stub)
