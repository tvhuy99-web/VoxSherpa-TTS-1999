#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def rewrite(path: str, old: str, new: str, count: int | None = None):
    p = ROOT / path
    s = p.read_text(encoding="utf-8")
    n = s.count(old)
    if n == 0:
        raise SystemExit(f"Anchor not found in {path}: {old[:120]!r}")
    if count is not None and n != count:
        raise SystemExit(f"Expected {count} matches in {path}, found {n}: {old[:120]!r}")
    p.write_text(s.replace(old, new), encoding="utf-8")
    print(f"patched {path}: {n} replacement(s)")


# Native QNN provider: reuse the proven QNN plumbing, target Qualcomm HTP/NPU,
# and force strict provider ownership so an "active HTP" result cannot silently
# include ORT CPU fallback nodes.
cpp = "app/src/main/cpp/kokoro_vi/kokoro_vi_jni.cpp"
rewrite(
    cpp,
    '''void append_qnn_gpu(Ort::SessionOptions& options) {
    const char* keys[] = {"backend_path"};
    const char* values[] = {"libQnnGpu.so"};
    const OrtApi& api = Ort::GetApi();
    OrtStatus* status = api.SessionOptionsAppendExecutionProvider(
            options, "QNN", keys, values, 1);
    if (status != nullptr) {
        const char* raw = api.GetErrorMessage(status);
        std::string message = raw == nullptr ? "unknown QNN GPU provider error" : raw;
        api.ReleaseStatus(status);
        throw std::runtime_error("Cannot enable QNN GPU: " + message);
    }
}
''',
    '''void append_qnn_gpu(Ort::SessionOptions& options) {
    const char* keys[] = {
            "backend_path",
            "htp_performance_mode",
            "htp_graph_finalization_optimization_mode"
    };
    const char* values[] = {
            "libQnnHtp.so",
            "burst",
            "3"
    };
    const OrtApi& api = Ort::GetApi();
    OrtStatus* status = api.SessionOptionsAppendExecutionProvider(
            options, "QNN", keys, values, 3);
    if (status != nullptr) {
        const char* raw = api.GetErrorMessage(status);
        std::string message = raw == nullptr ? "unknown QNN HTP provider error" : raw;
        api.ReleaseStatus(status);
        throw std::runtime_error("Cannot enable QNN HTP/NPU: " + message);
    }
}
''',
    1,
)
rewrite(
    cpp,
    '''    if (use_qnn_gpu) append_qnn_gpu(options);
    else if (use_nnapi) append_nnapi(options);
''',
    '''    if (use_qnn_gpu) {
        // Strict experiment: if any node cannot be assigned to QNN HTP, session
        // creation must fail instead of hiding CPU work inside an apparent HTP run.
        options.AddConfigEntry("session.disable_cpu_ep_fallback", "1");
        append_qnn_gpu(options);
    } else if (use_nnapi) append_nnapi(options);
''',
    1,
)
rewrite(
    cpp,
    '''        if (requested_qnn_gpu) {
            try {
                g_session = create_session(path, cpu_threads, false, true);
                g_active_qnn_gpu.store(true, std::memory_order_release);
                LOGI("Vietnamese Kokoro loaded with QNN GPU + cpuThreads=%d", cpu_threads);
            } catch (const std::exception& qnn_error) {
                LOGW("QNN GPU session failed, falling back to CPU: %s", qnn_error.what());
                g_session = create_session(path, cpu_threads, false, false);
                LOGI("Vietnamese Kokoro CPU fallback after QNN GPU failure, cpuThreads=%d", cpu_threads);
            }
        } else if (requested_nnapi) {
''',
    '''        if (requested_qnn_gpu) {
            // Do not CPU-fallback on the QDQ model here. Java owns the safe fallback
            // and will rebuild the original FP32 model if strict HTP cannot start.
            g_session = create_session(path, cpu_threads, false, true);
            g_active_qnn_gpu.store(true, std::memory_order_release);
            LOGI("Vietnamese Kokoro loaded with QNN HTP/NPU strict mode + cpuThreads=%d", cpu_threads);
        } else if (requested_nnapi) {
''',
    1,
)

# User/debug-facing native text only. Internal JNI method names deliberately remain
# qnnGpu to minimize experimental source churn.
for old, new in [
    ("QNN GPU +", "QNN HTP/NPU +"),
    ("QNN GPU session failed", "QNN HTP/NPU session failed"),
    ("after QNN GPU failure", "after QNN HTP/NPU failure"),
]:
    p = ROOT / cpp
    s = p.read_text(encoding="utf-8")
    if old in s:
        p.write_text(s.replace(old, new), encoding="utf-8")

# Native Java preload: the Qualcomm runtime AAR supplies the matching HTP dependency set.
native_java = "app/src/main/java/com/CodeBySonu/VoxSherpa/vietnamese/VietnameseKokoroNative.java"
rewrite(native_java, 'System.loadLibrary("QnnGpu");', 'System.loadLibrary("QnnHtp");', 1)
p = ROOT / native_java
s = p.read_text(encoding="utf-8")
s = s.replace("QNN GPU", "QNN HTP/NPU").replace("libQnnGpu", "libQnnHtp")
p.write_text(s, encoding="utf-8")

# Package and install a second model. CPU continues to use the untouched FP32 model;
# HTP uses the QNN U16/U8 QDQ model, making the A/B meaningful.
asset = "app/src/main/java/com/CodeBySonu/VoxSherpa/vietnamese/VietnameseKokoroAssetStore.java"
rewrite(asset,
        'import com.CodeBySonu.VoxSherpa.system.TtsDiagnostics;\n',
        'import com.CodeBySonu.VoxSherpa.BuildConfig;\nimport com.CodeBySonu.VoxSherpa.system.TtsDiagnostics;\n', 1)
rewrite(asset,
        '    private static final String MODEL = ROOT + "/kokoro_vi.onnx";\n',
        '    private static final String MODEL = ROOT + "/kokoro_vi.onnx";\n    private static final String QNN_MODEL = ROOT + "/kokoro_vi_qdq.onnx";\n', 1)
rewrite(asset,
        '''        public final File model;
        public final File dictionary;
        public final File voicepack;

        private Paths(File model, File dictionary, File voicepack) {
            this.model = model;
            this.dictionary = dictionary;
            this.voicepack = voicepack;
        }
''',
        '''        public final File model;
        public final File qnnModel;
        public final File dictionary;
        public final File voicepack;

        private Paths(File model, File qnnModel, File dictionary, File voicepack) {
            this.model = model;
            this.qnnModel = qnnModel;
            this.dictionary = dictionary;
            this.voicepack = voicepack;
        }
''', 1)
rewrite(asset,
        '''            boolean assets = assetExists(context, MODEL)
                    && assetExists(context, DICTIONARY)
                    && assetExists(context, CONFIG);
''',
        '''            boolean assets = assetExists(context, MODEL)
                    && assetExists(context, DICTIONARY)
                    && assetExists(context, CONFIG);
            if (BuildConfig.KOKORO_QNN_GPU_DEFAULT) {
                assets = assets && assetExists(context, QNN_MODEL);
            }
''', 1)
rewrite(asset,
        '''        File model = copyIfNeeded(context, MODEL, new File(root, "kokoro_vi.onnx"));
        File dictionary = copyIfNeeded(context, DICTIONARY, new File(root, "sea_g2p.bin"));
''',
        '''        File model = copyIfNeeded(context, MODEL, new File(root, "kokoro_vi.onnx"));
        File qnnModel = BuildConfig.KOKORO_QNN_GPU_DEFAULT
                ? copyIfNeeded(context, QNN_MODEL, new File(root, "kokoro_vi_qdq.onnx"))
                : null;
        File dictionary = copyIfNeeded(context, DICTIONARY, new File(root, "sea_g2p.bin"));
''', 1)
rewrite(asset,
        '''                "elapsedMs=" + elapsedMs + ", modelBytes=" + model.length()
                        + ", dictionaryBytes=" + dictionary.length()
''',
        '''                "elapsedMs=" + elapsedMs + ", modelBytes=" + model.length()
                        + ", qnnModelBytes=" + (qnnModel == null ? 0 : qnnModel.length())
                        + ", dictionaryBytes=" + dictionary.length()
''', 1)
rewrite(asset, '        return new Paths(model, dictionary, defaultVoice);\n',
              '        return new Paths(model, qnnModel, dictionary, defaultVoice);\n', 1)

engine = "app/src/main/java/com/CodeBySonu/VoxSherpa/vietnamese/VietnameseKokoroEngine.java"
rewrite(engine,
        '''            modelReady = nativeBridge.createEngine(
                    assets.model.getAbsolutePath(), activeCpuThreads, requestedNnapi, requestedQnnGpu);
''',
        '''            File selectedModel = requestedQnnGpu ? assets.qnnModel : assets.model;
            if (selectedModel == null || !selectedModel.isFile()) {
                throw new IllegalStateException("Requested Kokoro runtime model is missing: "
                        + (requestedQnnGpu ? "QNN HTP QDQ" : "CPU FP32"));
            }
            modelReady = nativeBridge.createEngine(
                    selectedModel.getAbsolutePath(), activeCpuThreads, requestedNnapi, requestedQnnGpu);
''', 1)
rewrite(engine,
        '"elapsedMs=" + elapsedMs(started) + ", modelBytes=" + assets.model.length()\n',
        '"elapsedMs=" + elapsedMs(started) + ", modelBytes=" + (requestedQnnGpu ? assets.qnnModel.length() : assets.model.length())\n', 1)
rewrite(engine,
        '''        prepare(app);
        if (!modelWarm && !warmCurrentSession(app, "runtime_backend_switch:" + requested)) {
            throw new IllegalStateException("Vietnamese Kokoro runtime switch warm-up was interrupted.");
        }

        String active = activeProviderLabel();
''',
        '''        try {
            prepare(app);
            if (!modelWarm && !warmCurrentSession(app, "runtime_backend_switch:" + requested)) {
                throw new IllegalStateException("Vietnamese Kokoro runtime switch warm-up was interrupted.");
            }
        } catch (Exception acceleratorError) {
            if (!BACKEND_QNN_GPU.equals(requested)) throw acceleratorError;
            TtsDiagnostics.warn(app, "provider", "qnn_htp_strict_failed",
                    "Strict QNN HTP session/inference failed; restoring original CPU FP32 model. error=" + acceleratorError);
            prefs.edit()
                    .putString(PREF_RUNTIME_BACKEND, BACKEND_CPU)
                    .putInt(PREF_CPU_THREADS, 0)
                    .putBoolean(PREF_NNAPI_ENABLED, false)
                    .apply();
            try { nativeBridge.destroyEngine(); } catch (Throwable ignored) {}
            modelReady = false;
            modelWarm = false;
            activeNnapi = false;
            activeQnnGpu = false;
            activeCpuThreads = 0;
            prepare(app);
            if (!modelWarm && !warmCurrentSession(app, "runtime_backend_strict_fallback_cpu_fp32")) {
                throw new IllegalStateException("CPU FP32 fallback warm-up was interrupted.");
            }
        }

        String active = activeProviderLabel();
''', 1)
p = ROOT / engine
s = p.read_text(encoding="utf-8")
s = s.replace('if (activeQnnGpu) return "QNN_GPU";', 'if (activeQnnGpu) return "QNN_HTP";')
s = s.replace("QNN GPU runtime is not packaged", "QNN HTP/NPU runtime is not packaged")
s = s.replace("QNN GPU experiment requested but libQnnGpu/libQnnSystem", "QNN HTP/NPU experiment requested but libQnnHtp/libQnnSystem")
s = s.replace("QNN GPU session could not be created", "QNN HTP/NPU session could not be created")
s = s.replace("native backend fell back to CPU", "strict HTP mode failed; Java will restore CPU FP32")
p.write_text(s, encoding="utf-8")

# Accessible diagnostics: preserve internal BACKEND_QNN_GPU preference value, but expose
# the actual backend to the user as strict HTP/NPU and state that CPU uses FP32 while HTP uses QDQ.
diag = "app/src/main/java/com/CodeBySonu/VoxSherpa/system/TtsDiagnosticsActivity.java"
p = ROOT / diag
s = p.read_text(encoding="utf-8")
s = s.replace("same ORT 1.26.0, same model and voice",
              "CPU FP32 vs Qualcomm HTP/NPU U16/U8 QDQ — same ORT 1.26.0 and voice")
s = s.replace("QNN GPU", "QNN HTP/NPU")
s = s.replace('if (gpu && !"QNN_GPU".equals(active))', 'if (gpu && !"QNN_HTP".equals(active))')
s = s.replace("same Kokoro model for a fair comparison with QNN HTP/NPU",
              "FP32 Kokoro model; HTP uses its U16/U8 QDQ counterpart in strict no-CPU-fallback mode")
s = s.replace("SELECTED / FALLBACK CPU", "REQUEST FAILED / CPU FP32 RESTORED")
p.write_text(s, encoding="utf-8")

print("QNN HTP/NPU strict integration patch complete")
