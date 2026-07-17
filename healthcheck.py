#!/usr/bin/env python3
"""Container healthcheck for the MiniMax-M3 SGLang server.

While the container is still starting (downloading weights and/or loading the
model), serve.sh keeps /tmp/.model-starting present and we report healthy, so a
legitimately long first boot is never marked unhealthy. serve.sh removes the flag
once /health first answers; from then on we probe /health directly.
"""
import os
import sys
import urllib.request

if os.path.exists("/tmp/.model-starting"):
    sys.exit(0)

port = os.environ.get("PORT", "30000")
try:
    with urllib.request.urlopen("http://127.0.0.1:%s/health" % port, timeout=8) as r:
        sys.exit(0 if r.status == 200 else 1)
except Exception:
    sys.exit(1)
