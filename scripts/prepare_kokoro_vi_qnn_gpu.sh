#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APP_MAIN="$ROOT/app/src/main"
JNI_DIR="$APP_MAIN/jniLibs"
CPP_DIR="$APP_MAIN/cpp/kokoro_vi"
WORK_DIR="${RUNNER_TEMP:-$ROOT/.kokoro-vi-work}"
ORT_QNN_VERSION="${ORT_QNN_VERSION:-1.26.0}"
QNN_RUNTIME_VERSION="${QNN_RUNTIME_VERSION:-2.42.0}"

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

# Reuse the proven Kokoro/sea-g2p/voicepack preparation. Its ORT binary is replaced
# below only for this QNN GPU experiment build.
ORT_VERSION=1.17.1 "$ROOT/scripts/prepare_kokoro_vi_stage1.sh"

ORT_QNN_AAR="$WORK_DIR/onnxruntime-android-qnn-${ORT_QNN_VERSION}.aar"
QNN_RUNTIME_AAR="$WORK_DIR/qnn-runtime-${QNN_RUNTIME_VERSION}.aar"
ORT_QNN_EXTRACT="$WORK_DIR/onnxruntime-qnn-${ORT_QNN_VERSION}"
QNN_RUNTIME_EXTRACT="$WORK_DIR/qnn-runtime-${QNN_RUNTIME_VERSION}"

fetch "https://repo1.maven.org/maven2/com/microsoft/onnxruntime/onnxruntime-android-qnn/${ORT_QNN_VERSION}/onnxruntime-android-qnn-${ORT_QNN_VERSION}.aar" \
  "$ORT_QNN_AAR"
fetch "https://repo1.maven.org/maven2/com/qualcomm/qti/qnn-runtime/${QNN_RUNTIME_VERSION}/qnn-runtime-${QNN_RUNTIME_VERSION}.aar" \
  "$QNN_RUNTIME_AAR"

rm -rf "$ORT_QNN_EXTRACT" "$QNN_RUNTIME_EXTRACT"
mkdir -p "$ORT_QNN_EXTRACT" "$QNN_RUNTIME_EXTRACT"
unzip -q "$ORT_QNN_AAR" -d "$ORT_QNN_EXTRACT"
unzip -q "$QNN_RUNTIME_AAR" -d "$QNN_RUNTIME_EXTRACT"

ORT_SO=$(find "$ORT_QNN_EXTRACT" -type f -path '*/arm64-v8a/libonnxruntime.so' | head -n 1)
test -n "$ORT_SO" || { echo "QNN AAR does not contain arm64 libonnxruntime.so" >&2; exit 1; }
cp "$ORT_SO" "$JNI_DIR/arm64-v8a/libonnxruntime.so"

rm -rf "$CPP_DIR/onnxruntime_headers"
mkdir -p "$CPP_DIR/onnxruntime_headers"
HEADER_ROOT=$(find "$ORT_QNN_EXTRACT" -type f -name onnxruntime_cxx_api.h -printf '%h\n' | head -n 1)
test -n "$HEADER_ROOT" || { echo "QNN AAR does not contain ONNX Runtime C/C++ headers" >&2; exit 1; }
cp -R "$HEADER_ROOT/." "$CPP_DIR/onnxruntime_headers/"

# qnn-runtime ships the Qualcomm backend dependency set. Copy the whole ARM64 set,
# not only libQnnGpu.so, so transitive backend/system dependencies stay coherent.
while IFS= read -r -d '' so; do
  cp "$so" "$JNI_DIR/arm64-v8a/$(basename "$so")"
done < <(find "$QNN_RUNTIME_EXTRACT" -type f -path '*/arm64-v8a/*.so' -print0)

# Some ORT QNN AAR revisions may ship an additional provider .so. Include any ARM64
# libraries from that AAR except libonnxruntime.so, which was copied explicitly above.
while IFS= read -r -d '' so; do
  name=$(basename "$so")
  [[ "$name" == "libonnxruntime.so" ]] && continue
  cp "$so" "$JNI_DIR/arm64-v8a/$name"
done < <(find "$ORT_QNN_EXTRACT" -type f -path '*/arm64-v8a/*.so' -print0)

test -s "$JNI_DIR/arm64-v8a/libonnxruntime.so"
test -s "$JNI_DIR/arm64-v8a/libsea_g2p_android.so"
test -s "$JNI_DIR/arm64-v8a/libQnnGpu.so"
test -s "$JNI_DIR/arm64-v8a/libQnnSystem.so"
test -s "$CPP_DIR/onnxruntime_headers/onnxruntime_cxx_api.h"
test -s "$CPP_DIR/onnxruntime_headers/onnxruntime_c_api.h"

printf '\nQNN GPU Kokoro experiment prepared.\n'
printf 'ORT QNN version: %s\n' "$ORT_QNN_VERSION"
printf 'QNN runtime:      %s\n' "$QNN_RUNTIME_VERSION"
printf 'ARM64 libraries:\n'
find "$JNI_DIR/arm64-v8a" -maxdepth 1 -type f -name '*.so' -printf '  %f\n' | sort
printf 'libonnxruntime.so SHA256: '
sha256sum "$JNI_DIR/arm64-v8a/libonnxruntime.so"
printf 'libQnnGpu.so SHA256:     '
sha256sum "$JNI_DIR/arm64-v8a/libQnnGpu.so"
