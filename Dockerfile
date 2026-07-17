# syntax=docker/dockerfile:1
#
# MiniMax-M3-uncensored-NVFP4 — SGLang serving image.
#
# Bakes the two ModelOpt / FlashInfer compatibility patches into the SGLang
# source tree so RunPod deployments need NO `-v` code mounts. Only the model
# weights are provided at run time (network volume or auto-download).
#
# The base is pinned by digest to the exact SGLang "dev-minimax-m3" build the
# model was validated against (SGLANG_BUILD_COMMIT 56e290315b8fdb4c8c10f8e31360d9bc3d878633,
# CUDA 13.0.1). Do NOT float this tag: the patches are engine-version-specific.
FROM lmsysorg/sglang@sha256:8cc6e6f90bf803e9817800b679173d0b526f2b42b2c61b7ecafecdadb610eb55

# --- Baked-in compatibility patches (previously mounted via -v) --------------
# From ressl/MiniMax-M3-uncensored-NVFP4:sglang_patch/, originally
# Mapika/MiniMax-M3-NVFP4 @ 668435825700a0047399441720f430bdd8eca0ab.
# They make SGLang's ModelOpt Cutlass path execute MiniMax-M3's parameterized
# clamped-SwiGLU correctly. Destinations mirror the validated mount targets.
COPY sglang_patch/modelopt_quant.py \
     /sgl-workspace/sglang/python/sglang/srt/layers/quantization/modelopt_quant.py
COPY sglang_patch/flashinfer_trtllm.py \
     /sgl-workspace/sglang/python/sglang/srt/layers/moe/moe_runner/flashinfer_trtllm.py

# Fail the build (not a 20-minute-in RunPod boot) if a patch file is truncated
# or corrupt. py_compile only parses/compiles — no torch or GPU needed.
RUN python3 -m py_compile \
      /sgl-workspace/sglang/python/sglang/srt/layers/quantization/modelopt_quant.py \
      /sgl-workspace/sglang/python/sglang/srt/layers/moe/moe_runner/flashinfer_trtllm.py

# Rust-based parallel Hub downloader — makes the optional ~260 GB first-boot
# fetch far faster and more robust. serve.sh enables it via HF_HUB_ENABLE_HF_TRANSFER.
RUN python3 -m pip install --no-cache-dir hf_transfer \
 || python3 -m pip install --no-cache-dir --break-system-packages hf_transfer

# --- Launch wrapper + healthcheck --------------------------------------------
COPY serve.sh healthcheck.py /usr/local/bin/
RUN chmod +x /usr/local/bin/serve.sh /usr/local/bin/healthcheck.py

LABEL org.opencontainers.image.title="MiniMax-M3-uncensored-NVFP4 SGLang server" \
      org.opencontainers.image.description="SGLang dev-minimax-m3 with ressl/MiniMax-M3-uncensored-NVFP4 ModelOpt+FlashInfer patches baked in" \
      org.opencontainers.image.base.name="lmsysorg/sglang@sha256:8cc6e6f90bf803e9817800b679173d0b526f2b42b2c61b7ecafecdadb610eb55" \
      com.ressl.model="ressl/MiniMax-M3-uncensored-NVFP4" \
      com.ressl.sglang-patch-source="Mapika/MiniMax-M3-NVFP4@668435825700a0047399441720f430bdd8eca0ab"

EXPOSE 30000

# SGLang exposes /health once the model is loaded. serve.sh keeps
# /tmp/.model-starting present from boot until /health first answers (covering the
# long download AND the multi-minute weight load), and healthcheck.py reports
# healthy while it exists — so no fixed start-period has to bound startup.
HEALTHCHECK --interval=30s --timeout=10s --start-period=1800s --retries=5 \
  CMD ["python3", "/usr/local/bin/healthcheck.py"]

# Keep the base NVIDIA entrypoint (/opt/nvidia/nvidia_entrypoint.sh); it execs CMD.
CMD ["/usr/local/bin/serve.sh"]
