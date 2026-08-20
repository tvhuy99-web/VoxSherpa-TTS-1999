#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APP_MAIN="$ROOT/app/src/main"
JNI_DIR="$APP_MAIN/jniLibs"
CPP_DIR="$APP_MAIN/cpp/kokoro_vi"
WORK_DIR="${RUNNER_TEMP:-$ROOT/.kokoro-vi-work}"
ORT_XNNPACK_VERSION="${ORT_XNNPACK_VERSION:-1.26.0}"

mkdir -p "$WORK_DIR" "$JNI_DIR/arm64-v8a"

fetch() {
  local url="$1"; local output="$2"
  if [[ -s "$output" ]]; then
    echo "Already present: $output"
    return
  fi
  echo "Downloading: $url"
  curl --fail --location --retry 4 --retry-delay 3 --connect-timeout 30 \
    --output "$output.part" "$url"
  mv "$output.part" "$output"
}

# Reuse the established Kokoro model, sea-g2p and 14 voicepack preparation.
# Replace only ONNX Runtime below so the XNNPACK experiment changes no speech assets.
ORT_VERSION=1.17.1 "$ROOT/scripts/prepare_kokoro_vi_stage1.sh"

ORT_AAR="$WORK_DIR/onnxruntime-android-${ORT_XNNPACK_VERSION}.aar"
ORT_EXTRACT="$WORK_DIR/onnxruntime-android-xnnpack-${ORT_XNNPACK_VERSION}"
fetch "https://repo1.maven.org/maven2/com/microsoft/onnxruntime/onnxruntime-android/${ORT_XNNPACK_VERSION}/onnxruntime-android-${ORT_XNNPACK_VERSION}.aar" \
  "$ORT_AAR"

rm -rf "$ORT_EXTRACT"
mkdir -p "$ORT_EXTRACT"
unzip -q "$ORT_AAR" -d "$ORT_EXTRACT"

ORT_SO=$(find "$ORT_EXTRACT" -type f -path '*/arm64-v8a/libonnxruntime.so' | head -n 1)
test -n "$ORT_SO" || { echo "ORT Android AAR does not contain arm64-v8a/libonnxruntime.so" >&2; exit 1; }
cp "$ORT_SO" "$JNI_DIR/arm64-v8a/libonnxruntime.so"

rm -rf "$CPP_DIR/onnxruntime_headers"
mkdir -p "$CPP_DIR/onnxruntime_headers"
HEADER_ROOT=$(find "$ORT_EXTRACT" -type f -name onnxruntime_cxx_api.h -printf '%h\n' | head -n 1)
test -n "$HEADER_ROOT" || { echo "ORT Android AAR does not contain ONNX Runtime C/C++ headers" >&2; exit 1; }
cp -R "$HEADER_ROOT/." "$CPP_DIR/onnxruntime_headers/"

# The experiment APK is ARM64-only; remove stale x86_64 payload created by shared prep.
rm -rf "$JNI_DIR/x86_64"

test -s "$JNI_DIR/arm64-v8a/libonnxruntime.so"
test -s "$JNI_DIR/arm64-v8a/libsea_g2p_android.so"
test -s "$CPP_DIR/onnxruntime_headers/onnxruntime_cxx_api.h"
test -s "$CPP_DIR/onnxruntime_headers/onnxruntime_c_api.h"

printf '\nXNNPACK Kokoro experiment prepared.\n'
printf 'ORT Android version: %s\n' "$ORT_XNNPACK_VERSION"
printf 'ABI:                 arm64-v8a\n'
printf 'libonnxruntime SHA256: '
sha256sum "$JNI_DIR/arm64-v8a/libonnxruntime.so"

# Official onnxruntime-android packages include XNNPACK EP. Keep a lightweight binary
# sanity check in CI; runtime diagnostics still decide whether the EP actually creates.
if strings "$JNI_DIR/arm64-v8a/libonnxruntime.so" | grep -E -m1 'XNNPACK|XnnpackExecutionProvider' >/dev/null; then
  echo 'XNNPACK marker found in libonnxruntime.so'
else
  echo 'WARNING: no textual XNNPACK marker found; compile/runtime checks will be authoritative.' >&2
fi
