# ──────────────────────────────────────────────
# FastAgents Bash sandbox      Tag: fastagents-bash:0.1
# ──────────────────────────────────────────────
FROM ubuntu:22.04

# Minimal packages: bash + basic GNU coreutils
RUN apt-get update \
 && apt-get install -y --no-install-recommends bash coreutils \
 && rm -rf /var/lib/apt/lists/*

# Non-root user (safety)
RUN useradd -ms /bin/bash agent
USER agent
WORKDIR /workspace
