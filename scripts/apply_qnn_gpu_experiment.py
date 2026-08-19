#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def replace_once(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    if old not in text:
        raise SystemExit(f"Expected text not found in {path}: {old[:160]!r}")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


engine = ROOT / "app/src/main/java/com/CodeBySonu/VoxSherpa/vietnamese/VietnameseKokoroEngine.java"
cpp = ROOT / "app/src/main/cpp/kokoro_vi/kokoro_vi_jni.cpp"

replace_once(engine,
    'import com.CodeBySonu.VoxSherpa.Sonic;\n',
    'import com.CodeBySonu.VoxSherpa.BuildConfig;\nimport com.CodeBySonu.VoxSherpa.Sonic;\n')

replace_once(engine,
    '    private volatile boolean activeNnapi;\n    private volatile int activeCpuThreads = 0;\n',
    '    private volatile boolean activeNnapi;\n    private volatile boolean activeQnnGpu;\n    private volatile int activeCpuThreads = 0;\n')

replace_once(engine,
'''    public boolean isNnapiActive() {
        return activeNnapi;
    }

    public String performanceState(Context context) {''',
'''    public boolean isNnapiActive() {
        return activeNnapi;
    }

    public boolean isQnnGpuActive() {
        return activeQnnGpu;
    }

    private String activeProviderLabel() {
        if (activeQnnGpu) return "QNN_GPU";
        if (activeNnapi) return "NNAPI";
        return "CPU";
    }

    public String performanceState(Context context) {''')

replace_once(engine,
'''            return "ready=" + isReady() + ", modelReady=" + modelReady + ", warm=" + modelWarm
                    + ", g2pReady=" + (g2pHandle != 0L) + ", provider=" + (activeNnapi ? "NNAPI" : "CPU")
                    + ", nnapiRequested=" + nnapiRequested + ", activeCpuThreads=" + activeCpuThreads
                    + ", savedCpuThreads=" + saved + " (0=ORT default/auto-affinity)"
                    + ", cachedVoiceStyles=" + voiceStyleCache.size();''',
'''            return "ready=" + isReady() + ", modelReady=" + modelReady + ", warm=" + modelWarm
                    + ", g2pReady=" + (g2pHandle != 0L) + ", provider=" + activeProviderLabel()
                    + ", qnnGpuBuild=" + BuildConfig.KOKORO_QNN_GPU_DEFAULT
                    + ", qnnGpuLibrary=" + VietnameseKokoroNative.isQnnGpuLibraryAvailable()
                    + ", nnapiRequested=" + nnapiRequested + ", activeCpuThreads=" + activeCpuThreads
                    + ", savedCpuThreads=" + saved + " (0=ORT default/auto-affinity)"
                    + ", cachedVoiceStyles=" + voiceStyleCache.size();''')

# Every explicit reset must clear the GPU provider state too.
text = engine.read_text(encoding="utf-8")
text = text.replace('        activeNnapi = false;\n', '        activeNnapi = false;\n        activeQnnGpu = false;\n')
engine.write_text(text, encoding="utf-8")

replace_once(engine,
'''            boolean requestedNnapi = prefs.getBoolean(PREF_NNAPI_ENABLED, false);
            modelReady = nativeBridge.createEngine(assets.model.getAbsolutePath(), activeCpuThreads, requestedNnapi);
            if (!modelReady) throw new IllegalStateException("Vietnamese Kokoro model failed to load.");
            activeNnapi = nativeBridge.isNnapiActive();
            if (requestedNnapi && !activeNnapi) {
                prefs.edit().putBoolean(PREF_NNAPI_ENABLED, false).apply();
                TtsDiagnostics.warn(context, "provider", "nnapi_auto_disabled",
                        "NNAPI session was unavailable or incompatible; continuing safely on CPU.");
            }
            TtsDiagnostics.info(context, "engine", "model_ready",
                    "elapsedMs=" + elapsedMs(started) + ", modelBytes=" + assets.model.length()
                            + ", provider=" + (activeNnapi ? "NNAPI" : "CPU")
                            + ", cpuThreads=" + threadLabel(activeCpuThreads));''',
'''            boolean requestedQnnGpu = BuildConfig.KOKORO_QNN_GPU_DEFAULT;
            boolean requestedNnapi = !requestedQnnGpu && prefs.getBoolean(PREF_NNAPI_ENABLED, false);
            if (requestedQnnGpu && !VietnameseKokoroNative.isQnnGpuLibraryAvailable()) {
                TtsDiagnostics.warn(context, "provider", "qnn_gpu_library_unavailable",
                        "QNN GPU experiment requested but libQnnGpu/libQnnSystem could not be preloaded: "
                                + VietnameseKokoroNative.qnnGpuLoadError());
            }
            modelReady = nativeBridge.createEngine(
                    assets.model.getAbsolutePath(), activeCpuThreads, requestedNnapi, requestedQnnGpu);
            if (!modelReady) throw new IllegalStateException("Vietnamese Kokoro model failed to load.");
            activeQnnGpu = nativeBridge.isQnnGpuActive();
            activeNnapi = nativeBridge.isNnapiActive();
            if (requestedQnnGpu && !activeQnnGpu) {
                TtsDiagnostics.warn(context, "provider", "qnn_gpu_fallback",
                        "QNN GPU session could not be created for this device/model; native backend fell back to CPU.");
            }
            if (requestedNnapi && !activeNnapi) {
                prefs.edit().putBoolean(PREF_NNAPI_ENABLED, false).apply();
                TtsDiagnostics.warn(context, "provider", "nnapi_auto_disabled",
                        "NNAPI session was unavailable or incompatible; continuing safely on CPU.");
            }
            TtsDiagnostics.info(context, "engine", "model_ready",
                    "elapsedMs=" + elapsedMs(started) + ", modelBytes=" + assets.model.length()
                            + ", provider=" + activeProviderLabel()
                            + ", qnnGpuRequested=" + requestedQnnGpu
                            + ", cpuThreads=" + threadLabel(activeCpuThreads));''')

# Profile and benchmark logs should report the actual active provider.
text = engine.read_text(encoding="utf-8")
text = text.replace('(activeNnapi ? "NNAPI" : "CPU")', 'activeProviderLabel()')
engine.write_text(text, encoding="utf-8")

# In a GPU experiment build, a manual CPU benchmark must restore the GPU session afterwards.
replace_once(engine,
    '        boolean restoreNnapi = prefs.getBoolean(PREF_NNAPI_ENABLED, false);\n',
    '        boolean restoreAccelerator = BuildConfig.KOKORO_QNN_GPU_DEFAULT\n                || prefs.getBoolean(PREF_NNAPI_ENABLED, false);\n')
replace_once(engine,
    '        if (restoreNnapi) {\n',
    '        if (restoreAccelerator) {\n')

# ---------------- Native QNN GPU provider ----------------
replace_once(cpp,
    'std::atomic<bool> g_active_nnapi{false};\n',
    'std::atomic<bool> g_active_nnapi{false};\nstd::atomic<bool> g_active_qnn_gpu{false};\n')

replace_once(cpp,
'''void append_nnapi(Ort::SessionOptions& options) {
    AppendNnapiFn append = nnapi_append_function();
    if (append == nullptr) {
        throw std::runtime_error("NNAPI execution provider is unavailable in this ONNX Runtime build.");
    }
    // ORT NNAPI flag 0x004 disables the NNAPI reference CPU device. Unsupported graph
    // portions can still fall back to ORT CPU, but we avoid routing supported nodes to
    // the usually-slower NNAPI CPU reference implementation.
    constexpr uint32_t NNAPI_FLAG_CPU_DISABLED = 0x004u;
    OrtStatus* status = append(options, NNAPI_FLAG_CPU_DISABLED);
    if (status != nullptr) {
        const OrtApi& api = Ort::GetApi();
        const char* raw = api.GetErrorMessage(status);
        std::string message = raw == nullptr ? "unknown NNAPI provider error" : raw;
        api.ReleaseStatus(status);
        throw std::runtime_error("Cannot enable NNAPI: " + message);
    }
}

std::unique_ptr<Ort::Session> create_session(const std::string& path, int cpu_threads, bool use_nnapi) {''',
'''void append_nnapi(Ort::SessionOptions& options) {
    AppendNnapiFn append = nnapi_append_function();
    if (append == nullptr) {
        throw std::runtime_error("NNAPI execution provider is unavailable in this ONNX Runtime build.");
    }
    // ORT NNAPI flag 0x004 disables the NNAPI reference CPU device. Unsupported graph
    // portions can still fall back to ORT CPU, but we avoid routing supported nodes to
    // the usually-slower NNAPI CPU reference implementation.
    constexpr uint32_t NNAPI_FLAG_CPU_DISABLED = 0x004u;
    OrtStatus* status = append(options, NNAPI_FLAG_CPU_DISABLED);
    if (status != nullptr) {
        const OrtApi& api = Ort::GetApi();
        const char* raw = api.GetErrorMessage(status);
        std::string message = raw == nullptr ? "unknown NNAPI provider error" : raw;
        api.ReleaseStatus(status);
        throw std::runtime_error("Cannot enable NNAPI: " + message);
    }
}

void append_qnn_gpu(Ort::SessionOptions& options) {
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

std::unique_ptr<Ort::Session> create_session(
        const std::string& path, int cpu_threads, bool use_nnapi, bool use_qnn_gpu) {''')

replace_once(cpp,
'''    options.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
    options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    if (use_nnapi) append_nnapi(options);
    return std::make_unique<Ort::Session>(g_env, path.c_str(), options);''',
'''    options.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
    options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    if (use_qnn_gpu) append_qnn_gpu(options);
    else if (use_nnapi) append_nnapi(options);
    return std::make_unique<Ort::Session>(g_env, path.c_str(), options);''')

replace_once(cpp,
'''extern "C" JNIEXPORT jboolean JNICALL Java_com_CodeBySonu_VoxSherpa_vietnamese_VietnameseKokoroNative_createEngine(
        JNIEnv* env, jobject, jstring model_path, jint cpu_threads, jboolean use_nnapi) {
    const auto path = jstring_to_utf8(env, model_path);
    const bool requested_nnapi = use_nnapi == JNI_TRUE;
    terminate_active_run();
    try {
        std::lock_guard<std::mutex> lock(g_mutex);
        g_session.reset();
        if (requested_nnapi) {
            try {
                g_session = create_session(path, cpu_threads, true);
                g_active_nnapi.store(true, std::memory_order_release);
                LOGI("Vietnamese Kokoro loaded with NNAPI + cpuThreads=%d", cpu_threads);
            } catch (const std::exception& nnapi_error) {
                LOGW("NNAPI session failed, falling back to CPU: %s", nnapi_error.what());
                g_session = create_session(path, cpu_threads, false);
                g_active_nnapi.store(false, std::memory_order_release);
                LOGI("Vietnamese Kokoro CPU fallback loaded with cpuThreads=%d", cpu_threads);
            }
        } else {
            g_session = create_session(path, cpu_threads, false);
            g_active_nnapi.store(false, std::memory_order_release);
            LOGI("Vietnamese Kokoro loaded with CPU cpuThreads=%d (0=ORT default + auto affinity)", cpu_threads);
        }
        g_active_threads = cpu_threads;
        return JNI_TRUE;''',
'''extern "C" JNIEXPORT jboolean JNICALL Java_com_CodeBySonu_VoxSherpa_vietnamese_VietnameseKokoroNative_createEngine(
        JNIEnv* env, jobject, jstring model_path, jint cpu_threads, jboolean use_nnapi, jboolean use_qnn_gpu) {
    const auto path = jstring_to_utf8(env, model_path);
    const bool requested_nnapi = use_nnapi == JNI_TRUE;
    const bool requested_qnn_gpu = use_qnn_gpu == JNI_TRUE;
    terminate_active_run();
    try {
        std::lock_guard<std::mutex> lock(g_mutex);
        g_session.reset();
        g_active_nnapi.store(false, std::memory_order_release);
        g_active_qnn_gpu.store(false, std::memory_order_release);
        if (requested_qnn_gpu) {
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
            try {
                g_session = create_session(path, cpu_threads, true, false);
                g_active_nnapi.store(true, std::memory_order_release);
                LOGI("Vietnamese Kokoro loaded with NNAPI + cpuThreads=%d", cpu_threads);
            } catch (const std::exception& nnapi_error) {
                LOGW("NNAPI session failed, falling back to CPU: %s", nnapi_error.what());
                g_session = create_session(path, cpu_threads, false, false);
                LOGI("Vietnamese Kokoro CPU fallback loaded with cpuThreads=%d", cpu_threads);
            }
        } else {
            g_session = create_session(path, cpu_threads, false, false);
            LOGI("Vietnamese Kokoro loaded with CPU cpuThreads=%d (0=ORT default + auto affinity)", cpu_threads);
        }
        g_active_threads = cpu_threads;
        return JNI_TRUE;''')

# Clear QNN state alongside NNAPI and update all CPU-only session creation call sites.
text = cpp.read_text(encoding="utf-8")
text = text.replace('g_active_nnapi.store(false, std::memory_order_release);\n',
                    'g_active_nnapi.store(false, std::memory_order_release);\n        g_active_qnn_gpu.store(false, std::memory_order_release);\n')
# The replacement above also touched places with different indentation; normalize harmless over-indentation later.
text = text.replace('                g_active_qnn_gpu.store(false, std::memory_order_release);\n        g_active_qnn_gpu.store(false, std::memory_order_release);\n',
                    '                g_active_qnn_gpu.store(false, std::memory_order_release);\n')
text = text.replace('g_active_qnn_gpu.store(false, std::memory_order_release);\n        g_active_qnn_gpu.store(false, std::memory_order_release);\n',
                    'g_active_qnn_gpu.store(false, std::memory_order_release);\n')
text = text.replace('create_session(path, candidate, false)', 'create_session(path, candidate, false, false)')
text = text.replace('create_session(path, best_threads, false)', 'create_session(path, best_threads, false, false)')
cpp.write_text(text, encoding="utf-8")

replace_once(cpp,
'''extern "C" JNIEXPORT jboolean JNICALL Java_com_CodeBySonu_VoxSherpa_vietnamese_VietnameseKokoroNative_isNnapiActive(JNIEnv*, jobject) {
    return g_active_nnapi.load(std::memory_order_acquire) ? JNI_TRUE : JNI_FALSE;
}

extern "C" JNIEXPORT jfloatArray''',
'''extern "C" JNIEXPORT jboolean JNICALL Java_com_CodeBySonu_VoxSherpa_vietnamese_VietnameseKokoroNative_isNnapiActive(JNIEnv*, jobject) {
    return g_active_nnapi.load(std::memory_order_acquire) ? JNI_TRUE : JNI_FALSE;
}

extern "C" JNIEXPORT jboolean JNICALL Java_com_CodeBySonu_VoxSherpa_vietnamese_VietnameseKokoroNative_isQnnGpuActive(JNIEnv*, jobject) {
    return g_active_qnn_gpu.load(std::memory_order_acquire) ? JNI_TRUE : JNI_FALSE;
}

extern "C" JNIEXPORT jfloatArray''')

print("QNN GPU experiment patches applied")
