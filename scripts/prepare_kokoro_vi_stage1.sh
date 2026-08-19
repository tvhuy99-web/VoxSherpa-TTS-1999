#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APP_MAIN="$ROOT/app/src/main"
ASSET_DIR="$APP_MAIN/assets/kokoro_vi"
JNI_DIR="$APP_MAIN/jniLibs"
CPP_DIR="$APP_MAIN/cpp/kokoro_vi"
TOOLS_DIR="$ROOT/tools/kokoro_vi/sea_g2p_android"
WORK_DIR="${RUNNER_TEMP:-$ROOT/.kokoro-vi-work}"
HF_REPO="contextboxai/Kokoro-Vietnamese"
HF_REVISION="9f210d622209fcc216fe2ac6159fed2ff381cb8a"
SEA_G2P_COMMIT="59001f6dc3ba729a4fb7c7d81f262b7447a68c21"
ORT_VERSION="${ORT_VERSION:-1.28.0}"
VOICES=(
  diem_trinh hung_thinh mai_linh mai_loan manh_dung my_yen ngoc_huyen
  phat_tai thanh_dat thuc_trinh tuan_ngoc storyvert duc_an duc_duy
)
mkdir -p "$ASSET_DIR/voicepacks" "$JNI_DIR/arm64-v8a" "$JNI_DIR/x86_64" "$WORK_DIR"

fetch() {
  local url="$1"; local output="$2"
  if [[ -s "$output" ]]; then echo "Already present: $output"; return; fi
  echo "Downloading: $url"
  curl --fail --location --retry 4 --retry-delay 3 --connect-timeout 30 --output "$output.part" "$url"
  mv "$output.part" "$output"
}

MODEL="$ASSET_DIR/kokoro_vi.onnx"
DICTIONARY="$ASSET_DIR/sea_g2p.bin"
fetch "https://huggingface.co/${HF_REPO}/resolve/${HF_REVISION}/kokoro_vi.onnx?download=true" "$MODEL"
fetch "https://raw.githubusercontent.com/pnnbao97/sea-g2p/${SEA_G2P_COMMIT}/python/sea_g2p/sea_g2p.bin" "$DICTIONARY"

for voice in "${VOICES[@]}"; do
  voice_pt="$WORK_DIR/${voice}.pt"
  voice_f32="$ASSET_DIR/voicepacks/${voice}.f32le"
  fetch "https://huggingface.co/${HF_REPO}/resolve/${HF_REVISION}/voicepacks/${voice}.pt?download=true" "$voice_pt"
  python3 "$ROOT/tools/kokoro_vi/convert_voicepack.py" "$voice_pt" "$voice_f32"
  voice_size=$(wc -c < "$voice_f32")
  (( voice_size == 522240 )) || { echo "Unexpected ${voice} voicepack size: $voice_size" >&2; exit 1; }
done

model_size=$(wc -c < "$MODEL"); dict_size=$(wc -c < "$DICTIONARY")
(( model_size >= 300000000 )) || { echo "kokoro_vi.onnx is unexpectedly small: $model_size" >&2; exit 1; }
(( dict_size >= 50000000 )) || { echo "sea_g2p.bin is unexpectedly small: $dict_size" >&2; exit 1; }
test -s "$ASSET_DIR/config.json"

# Use Microsoft's official Maven-published Android AAR. The AAR contains both
# libonnxruntime.so and the public C/C++ headers needed by our JNI bridge.
ORT_AAR="$WORK_DIR/onnxruntime-android-${ORT_VERSION}.aar"
ORT_EXTRACT="$WORK_DIR/onnxruntime-${ORT_VERSION}"
fetch "https://repo1.maven.org/maven2/com/microsoft/onnxruntime/onnxruntime-android/${ORT_VERSION}/onnxruntime-android-${ORT_VERSION}.aar" "$ORT_AAR"
rm -rf "$ORT_EXTRACT"; mkdir -p "$ORT_EXTRACT"; unzip -q "$ORT_AAR" -d "$ORT_EXTRACT"
cp "$ORT_EXTRACT/jni/arm64-v8a/libonnxruntime.so" "$JNI_DIR/arm64-v8a/libonnxruntime.so"
cp "$ORT_EXTRACT/jni/x86_64/libonnxruntime.so" "$JNI_DIR/x86_64/libonnxruntime.so"
rm -rf "$CPP_DIR/onnxruntime_headers"; mkdir -p "$CPP_DIR/onnxruntime_headers"
cp -R "$ORT_EXTRACT/headers/." "$CPP_DIR/onnxruntime_headers/"

SEA_ARCHIVE="$WORK_DIR/sea-g2p-${SEA_G2P_COMMIT}.tar.gz"
SEA_SOURCE="$WORK_DIR/sea-g2p-src"
fetch "https://github.com/pnnbao97/sea-g2p/archive/${SEA_G2P_COMMIT}.tar.gz" "$SEA_ARCHIVE"
rm -rf "$SEA_SOURCE"; mkdir -p "$SEA_SOURCE"
tar -xzf "$SEA_ARCHIVE" --strip-components=1 -C "$SEA_SOURCE"

# Android needs only the native Rust core. Strip Python-only pyo3 annotations while
# preserving upstream Vietnamese normalization/G2P behavior at the pinned commit.
VI_MOD="$SEA_SOURCE/src/lang/vi/mod.rs"
python3 - "$VI_MOD" <<'PY'
from pathlib import Path
import re
import sys

path = Path(sys.argv[1])
text = path.read_text(encoding="utf-8")
text = text.replace("use pyo3::prelude::*;\n", "")
text = re.sub(r"(?m)^\s*#\[(?:pyclass|pymethods|new)\]\s*\n", "", text)
text = re.sub(r"(?m)^\s*#\[pyo3\([^\n]*\)\]\s*\n", "", text)
text = text.replace(
    "    pub fn normalize_batch(&self, py: Python<'_>, texts: Vec<String>, punc_norm: bool) -> PyResult<Vec<String>> {\n"
    "        py.allow_threads(|| {\n"
    "            use rayon::prelude::*;\n"
    "            Ok(texts.into_par_iter().map(|t| self.normalize(&t, punc_norm)).collect())\n"
    "        })\n"
    "    }",
    "    pub fn normalize_batch(&self, texts: Vec<String>, punc_norm: bool) -> Vec<String> {\n"
    "        use rayon::prelude::*;\n"
    "        texts.into_par_iter().map(|t| self.normalize(&t, punc_norm)).collect()\n"
    "    }",
)
if "pyo3::" in text or "#[pyo3" in text or "Python<'_>" in text or "PyResult<" in text:
    raise SystemExit("Python-only pyo3 bindings remain in Vietnamese module")
path.write_text(text, encoding="utf-8")
PY

cp "$TOOLS_DIR/Cargo.toml" "$SEA_SOURCE/Cargo.toml"
cp "$TOOLS_DIR/lib.rs" "$SEA_SOURCE/src/lib.rs"
rm -f "$SEA_SOURCE/Cargo.lock"
command -v cargo-ndk >/dev/null 2>&1 || { echo "cargo-ndk is required" >&2; exit 1; }
pushd "$SEA_SOURCE" >/dev/null
cargo ndk -t arm64-v8a -t x86_64 -o "$JNI_DIR" build --release
popd >/dev/null

for abi in arm64-v8a x86_64; do
  test -s "$JNI_DIR/$abi/libsea_g2p_android.so"
  test -s "$JNI_DIR/$abi/libonnxruntime.so"
done

test -s "$CPP_DIR/onnxruntime_headers/onnxruntime_cxx_api.h"
test -s "$CPP_DIR/onnxruntime_headers/onnxruntime_c_api.h"

printf '\nVietnamese Kokoro assets prepared: %d voices.\n' "${#VOICES[@]}"
printf 'ONNX Runtime     %s\n' "$ORT_VERSION"
printf 'kokoro_vi.onnx  '; sha256sum "$MODEL"
printf 'sea_g2p.bin     '; sha256sum "$DICTIONARY"
for voice in "${VOICES[@]}"; do
  printf '%-18s ' "${voice}.f32le"
  sha256sum "$ASSET_DIR/voicepacks/${voice}.f32le"
done
