#include <jni.h>
#include <android/log.h>
#include <onnxruntime_cxx_api.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <dlfcn.h>
#include <limits>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#define LOG_TAG "VoxSherpaKokoroVI"
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN, LOG_TAG, __VA_ARGS__)
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)

extern "C" {
void* sea_g2p_create(const char* dictionary_path);
char* sea_g2p_phonemize(void* handle, const char* text);
void sea_g2p_free_string(char* value);
void sea_g2p_destroy(void* handle);
}

namespace {
using Clock = std::chrono::steady_clock;
using AppendNnapiFn = OrtStatus* (*)(OrtSessionOptions*, uint32_t);

Ort::Env g_env(ORT_LOGGING_LEVEL_WARNING, "VoxSherpaKokoroVietnamese");
std::unique_ptr<Ort::Session> g_session;
std::mutex g_mutex;
std::mutex g_run_options_mutex;
Ort::RunOptions* g_active_run_options = nullptr;
std::atomic<bool> g_cancel_requested{false};
std::atomic<bool> g_active_nnapi{false};
std::atomic<bool> g_active_xnnpack{false};
std::array<int64_t, 7> g_last_timing_us{0,0,0,0,0,0,0};
int g_active_threads = 0;
int g_active_xnnpack_threads = 0;

int64_t us_since(const Clock::time_point& start) {
    return std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - start).count();
}

void throw_runtime(JNIEnv* env, const std::string& message) {
    jclass cls = env->FindClass("java/lang/IllegalStateException");
    if (cls != nullptr) env->ThrowNew(cls, message.c_str());
}

std::string jstring_to_utf8(JNIEnv* env, jstring value) {
    if (value == nullptr) return {};
    const char* chars = env->GetStringUTFChars(value, nullptr);
    if (chars == nullptr) return {};
    std::string result(chars);
    env->ReleaseStringUTFChars(value, chars);
    return result;
}

AppendNnapiFn nnapi_append_function() {
    static std::once_flag once;
    static AppendNnapiFn function = nullptr;
    static void* ort_handle = nullptr;
    std::call_once(once, []() {
        ort_handle = dlopen("libonnxruntime.so", RTLD_NOW | RTLD_LOCAL);
        if (ort_handle != nullptr) {
            function = reinterpret_cast<AppendNnapiFn>(
                    dlsym(ort_handle, "OrtSessionOptionsAppendExecutionProvider_Nnapi"));
        }
        if (function == nullptr) {
            LOGW("NNAPI provider factory symbol is unavailable in this ONNX Runtime build");
        }
    });
    return function;
}

void append_nnapi(Ort::SessionOptions& options) {
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

void append_xnnpack(Ort::SessionOptions& options, int xnnpack_threads) {
    const std::string thread_count = std::to_string(std::max(1, xnnpack_threads));
    const char* keys[] = {"intra_op_num_threads"};
    const char* values[] = {thread_count.c_str()};
    const OrtApi& api = Ort::GetApi();
    OrtStatus* status = api.SessionOptionsAppendExecutionProvider(
            options, "XNNPACK", keys, values, 1);
    if (status != nullptr) {
        const char* raw = api.GetErrorMessage(status);
        std::string message = raw == nullptr ? "unknown XNNPACK provider error" : raw;
        api.ReleaseStatus(status);
        throw std::runtime_error("Cannot enable XNNPACK: " + message);
    }
}

std::unique_ptr<Ort::Session> create_session(
        const std::string& path, int cpu_threads, bool use_nnapi, bool use_xnnpack) {
    Ort::SessionOptions options;
    int xnnpack_threads = 0;
    if (use_xnnpack) {
        const unsigned int detected = std::thread::hardware_concurrency();
        xnnpack_threads = std::max(1, std::min(8, static_cast<int>(detected == 0 ? 4u : detected)));
        // XNNPACK owns a separate intra-op pool. ORT recommends a single ORT intra-op
        // thread plus disabled spinning to avoid thread-pool contention.
        options.SetIntraOpNumThreads(1);
        options.AddConfigEntry("session.intra_op.allow_spinning", "0");
    } else if (cpu_threads > 0) {
        options.SetIntraOpNumThreads(cpu_threads);
    }
    options.SetInterOpNumThreads(1);
    options.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
    options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    if (use_xnnpack) append_xnnpack(options, xnnpack_threads);
    else if (use_nnapi) append_nnapi(options);
    return std::make_unique<Ort::Session>(g_env, path.c_str(), options);
}

bool terminate_active_run() {
    std::lock_guard<std::mutex> run_lock(g_run_options_mutex);
    if (g_active_run_options == nullptr) return false;
    g_cancel_requested.store(true, std::memory_order_release);
    try {
        g_active_run_options->SetTerminate();
        return true;
    } catch (const std::exception& e) {
        LOGW("SetTerminate failed: %s", e.what());
        return false;
    }
}

class ActiveRunScope {
public:
    explicit ActiveRunScope(Ort::RunOptions& options) : options_(&options) {
        g_cancel_requested.store(false, std::memory_order_release);
        std::lock_guard<std::mutex> run_lock(g_run_options_mutex);
        g_active_run_options = options_;
    }

    ~ActiveRunScope() {
        std::lock_guard<std::mutex> run_lock(g_run_options_mutex);
        if (g_active_run_options == options_) g_active_run_options = nullptr;
    }

private:
    Ort::RunOptions* options_;
};

struct RunResult {
    int64_t elapsed_us = 0;
    size_t samples = 0;
};

RunResult run_once(Ort::Session& session, const std::vector<int64_t>& ids,
                   const std::vector<float>& style, float speed) {
    auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    const std::array<int64_t,2> ids_shape{1, static_cast<int64_t>(ids.size())};
    const std::array<int64_t,2> style_shape{1, 256};
    const std::array<int64_t,0> scalar_shape{};
    auto ids_tensor = Ort::Value::CreateTensor<int64_t>(memory, const_cast<int64_t*>(ids.data()), ids.size(), ids_shape.data(), ids_shape.size());
    auto style_tensor = Ort::Value::CreateTensor<float>(memory, const_cast<float*>(style.data()), style.size(), style_shape.data(), style_shape.size());
    auto speed_tensor = Ort::Value::CreateTensor<float>(memory, &speed, 1, scalar_shape.data(), scalar_shape.size());
    const char* input_names[] = {"input_ids", "ref_s", "speed"};
    const char* output_names[] = {"waveform"};
    std::array<Ort::Value,3> inputs{std::move(ids_tensor), std::move(style_tensor), std::move(speed_tensor)};
    auto started = Clock::now();
    auto outputs = session.Run(Ort::RunOptions{nullptr}, input_names, inputs.data(), inputs.size(), output_names, 1);
    RunResult r;
    r.elapsed_us = us_since(started);
    r.samples = outputs[0].GetTensorTypeAndShapeInfo().GetElementCount();
    return r;
}

int64_t median_us(std::vector<int64_t> values) {
    if (values.empty()) return 0;
    std::sort(values.begin(), values.end());
    const size_t mid = values.size() / 2;
    if ((values.size() & 1u) != 0u) return values[mid];
    return (values[mid - 1] + values[mid]) / 2;
}

std::vector<int64_t> resize_ids(const std::vector<int64_t>& base, size_t requested_size) {
    if (base.size() < 3) return base;
    const size_t target_size = std::max<size_t>(3, std::min<size_t>(512, requested_size));
    std::vector<int64_t> out;
    out.reserve(target_size);
    out.push_back(base.front());
    const size_t body_begin = 1;
    const size_t body_end = base.size() - 1;
    size_t cursor = body_begin;
    while (out.size() + 1 < target_size) {
        out.push_back(base[cursor]);
        cursor++;
        if (cursor >= body_end) cursor = body_begin;
    }
    out.push_back(base.back());
    return out;
}

std::vector<int> cpu_thread_candidates(unsigned int online_cpus) {
    const unsigned int capped = std::max(1u, std::min(8u, online_cpus));
    std::vector<int> result;
    result.reserve(static_cast<size_t>(capped) + 1);
    result.push_back(0);
    for (unsigned int threads = 1; threads <= capped; ++threads) {
        result.push_back(static_cast<int>(threads));
    }
    return result;
}

struct CandidateBenchmark {
    int threads = 0;
    int64_t load_ms = 0;
    std::array<int64_t,3> median_us{0,0,0};
    int64_t score_us = 0;
};
}

extern "C" JNIEXPORT jlong JNICALL Java_com_CodeBySonu_VoxSherpa_vietnamese_VietnameseKokoroNative_createG2p(JNIEnv* env, jobject, jstring dictionary_path) {
    const auto path = jstring_to_utf8(env, dictionary_path);
    void* handle = sea_g2p_create(path.c_str());
    if (handle == nullptr) throw_runtime(env, "Cannot initialize Vietnamese sea-g2p.");
    return reinterpret_cast<jlong>(handle);
}

extern "C" JNIEXPORT jstring JNICALL Java_com_CodeBySonu_VoxSherpa_vietnamese_VietnameseKokoroNative_phonemize(JNIEnv* env, jobject, jlong handle, jstring text) {
    if (handle == 0L) { throw_runtime(env, "Vietnamese sea-g2p is not initialized."); return nullptr; }
    const auto input = jstring_to_utf8(env, text);
    char* output = sea_g2p_phonemize(reinterpret_cast<void*>(handle), input.c_str());
    if (output == nullptr) { throw_runtime(env, "Vietnamese sea-g2p failed."); return nullptr; }
    jstring result = env->NewStringUTF(output);
    sea_g2p_free_string(output);
    return result;
}

extern "C" JNIEXPORT void JNICALL Java_com_CodeBySonu_VoxSherpa_vietnamese_VietnameseKokoroNative_destroyG2p(JNIEnv*, jobject, jlong handle) {
    if (handle != 0L) sea_g2p_destroy(reinterpret_cast<void*>(handle));
}

extern "C" JNIEXPORT jboolean JNICALL Java_com_CodeBySonu_VoxSherpa_vietnamese_VietnameseKokoroNative_createEngine(
        JNIEnv* env, jobject, jstring model_path, jint cpu_threads, jboolean use_nnapi, jboolean use_xnnpack) {
    const auto path = jstring_to_utf8(env, model_path);
    const bool requested_xnnpack = use_xnnpack == JNI_TRUE;
    const bool requested_nnapi = !requested_xnnpack && use_nnapi == JNI_TRUE;
    terminate_active_run();
    try {
        std::lock_guard<std::mutex> lock(g_mutex);
        g_session.reset();
        g_active_nnapi.store(false, std::memory_order_release);
        g_active_xnnpack.store(false, std::memory_order_release);
        g_active_xnnpack_threads = 0;
        if (requested_xnnpack) {
            try {
                const unsigned int detected = std::thread::hardware_concurrency();
                const int xnn_threads = std::max(1, std::min(8, static_cast<int>(detected == 0 ? 4u : detected)));
                g_session = create_session(path, 1, false, true);
                g_active_xnnpack.store(true, std::memory_order_release);
                g_active_threads = 1;
                g_active_xnnpack_threads = xnn_threads;
                LOGI("Vietnamese Kokoro loaded with XNNPACK, ORT intra-op=1, xnnpackThreads=%d", xnn_threads);
            } catch (const std::exception& xnn_error) {
                LOGW("XNNPACK session failed, falling back to CPU: %s", xnn_error.what());
                g_session = create_session(path, cpu_threads, false, false);
                g_active_threads = cpu_threads;
                LOGI("Vietnamese Kokoro CPU fallback after XNNPACK failure, cpuThreads=%d", cpu_threads);
            }
        } else if (requested_nnapi) {
            try {
                g_session = create_session(path, cpu_threads, true, false);
                g_active_nnapi.store(true, std::memory_order_release);
                g_active_threads = cpu_threads;
                LOGI("Vietnamese Kokoro loaded with NNAPI + cpuThreads=%d", cpu_threads);
            } catch (const std::exception& nnapi_error) {
                LOGW("NNAPI session failed, falling back to CPU: %s", nnapi_error.what());
                g_session = create_session(path, cpu_threads, false, false);
                g_active_threads = cpu_threads;
                LOGI("Vietnamese Kokoro CPU fallback loaded with cpuThreads=%d", cpu_threads);
            }
        } else {
            g_session = create_session(path, cpu_threads, false, false);
            g_active_threads = cpu_threads;
            LOGI("Vietnamese Kokoro loaded with CPU cpuThreads=%d (0=ORT default + auto affinity)", cpu_threads);
        }
        return JNI_TRUE;
    } catch (const Ort::Exception& e) {
        g_active_nnapi.store(false, std::memory_order_release);
        g_active_xnnpack.store(false, std::memory_order_release);
        g_active_xnnpack_threads = 0;
        LOGE("ONNX load error: %s", e.what());
        throw_runtime(env, std::string("Vietnamese Kokoro model load failed: ") + e.what());
    } catch (const std::exception& e) {
        g_active_nnapi.store(false, std::memory_order_release);
        g_active_xnnpack.store(false, std::memory_order_release);
        g_active_xnnpack_threads = 0;
        throw_runtime(env, std::string("Vietnamese Kokoro native load failed: ") + e.what());
    }
    return JNI_FALSE;
}

extern "C" JNIEXPORT void JNICALL Java_com_CodeBySonu_VoxSherpa_vietnamese_VietnameseKokoroNative_destroyEngine(JNIEnv*, jobject) {
    terminate_active_run();
    std::lock_guard<std::mutex> lock(g_mutex);
    g_session.reset();
    g_active_nnapi.store(false, std::memory_order_release);
    g_active_xnnpack.store(false, std::memory_order_release);
    g_active_xnnpack_threads = 0;
}

extern "C" JNIEXPORT jboolean JNICALL Java_com_CodeBySonu_VoxSherpa_vietnamese_VietnameseKokoroNative_cancelActiveRun(JNIEnv*, jobject) {
    const bool terminated = terminate_active_run();
    if (terminated) LOGI("Active ONNX RunOptions terminated");
    return terminated ? JNI_TRUE : JNI_FALSE;
}

extern "C" JNIEXPORT jboolean JNICALL Java_com_CodeBySonu_VoxSherpa_vietnamese_VietnameseKokoroNative_isNnapiActive(JNIEnv*, jobject) {
    return g_active_nnapi.load(std::memory_order_acquire) ? JNI_TRUE : JNI_FALSE;
}

extern "C" JNIEXPORT jboolean JNICALL Java_com_CodeBySonu_VoxSherpa_vietnamese_VietnameseKokoroNative_isXnnpackActive(JNIEnv*, jobject) {
    return g_active_xnnpack.load(std::memory_order_acquire) ? JNI_TRUE : JNI_FALSE;
}

extern "C" JNIEXPORT jfloatArray JNICALL Java_com_CodeBySonu_VoxSherpa_vietnamese_VietnameseKokoroNative_synthesize(JNIEnv* env, jobject, jlongArray input_ids, jfloatArray ref_style, jfloat speed) {
    auto total_start = Clock::now();
    auto lock_start = Clock::now();
    std::unique_lock<std::mutex> lock(g_mutex);
    const int64_t lock_wait_us = us_since(lock_start);
    try {
        if (!g_session) { throw_runtime(env, "Vietnamese Kokoro model is not loaded."); return nullptr; }
        auto prep_start = Clock::now();
        const jsize id_count = env->GetArrayLength(input_ids);
        const jsize style_count = env->GetArrayLength(ref_style);
        if (id_count < 3 || id_count > 512 || style_count != 256) { throw_runtime(env, "Invalid Vietnamese Kokoro input."); return nullptr; }
        jlong* java_ids = env->GetLongArrayElements(input_ids, nullptr);
        jfloat* java_style = env->GetFloatArrayElements(ref_style, nullptr);
        if (java_ids == nullptr || java_style == nullptr) {
            if (java_ids) env->ReleaseLongArrayElements(input_ids, java_ids, JNI_ABORT);
            if (java_style) env->ReleaseFloatArrayElements(ref_style, java_style, JNI_ABORT);
            throw_runtime(env, "Not enough memory for Vietnamese Kokoro input."); return nullptr;
        }
        std::vector<int64_t> ids(static_cast<size_t>(id_count));
        for (jsize i = 0; i < id_count; ++i) ids[static_cast<size_t>(i)] = static_cast<int64_t>(java_ids[i]);
        std::vector<float> style(java_style, java_style + style_count);
        env->ReleaseLongArrayElements(input_ids, java_ids, JNI_ABORT);
        env->ReleaseFloatArrayElements(ref_style, java_style, JNI_ABORT);
        const int64_t input_prep_us = us_since(prep_start);

        auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        const std::array<int64_t,2> ids_shape{1, static_cast<int64_t>(id_count)};
        const std::array<int64_t,2> style_shape{1, 256};
        const std::array<int64_t,0> scalar_shape{};
        auto ids_tensor = Ort::Value::CreateTensor<int64_t>(memory, ids.data(), ids.size(), ids_shape.data(), ids_shape.size());
        auto style_tensor = Ort::Value::CreateTensor<float>(memory, style.data(), style.size(), style_shape.data(), style_shape.size());
        auto speed_tensor = Ort::Value::CreateTensor<float>(memory, &speed, 1, scalar_shape.data(), scalar_shape.size());
        const char* input_names[] = {"input_ids", "ref_s", "speed"};
        const char* output_names[] = {"waveform"};
        std::array<Ort::Value,3> inputs{std::move(ids_tensor), std::move(style_tensor), std::move(speed_tensor)};

        Ort::RunOptions run_options;
        ActiveRunScope active_run(run_options);
        auto ort_start = Clock::now();
        std::vector<Ort::Value> outputs;
        try {
            outputs = g_session->Run(run_options, input_names, inputs.data(), inputs.size(), output_names, 1);
        } catch (const Ort::Exception& e) {
            const int64_t cancelled_run_us = us_since(ort_start);
            if (g_cancel_requested.load(std::memory_order_acquire)) {
                g_last_timing_us = {lock_wait_us, input_prep_us, cancelled_run_us, 0, us_since(total_start), g_active_threads, g_active_xnnpack_threads};
                LOGI("ONNX synthesis terminated after %lld us", static_cast<long long>(cancelled_run_us));
                return nullptr;
            }
            throw;
        }
        const int64_t ort_run_us = us_since(ort_start);

        auto copy_start = Clock::now();
        const float* audio = outputs[0].GetTensorData<float>();
        const size_t sample_count = outputs[0].GetTensorTypeAndShapeInfo().GetElementCount();
        if (sample_count == 0 || sample_count > 12000000) { throw_runtime(env, "Vietnamese Kokoro returned invalid audio."); return nullptr; }
        jfloatArray result = env->NewFloatArray(static_cast<jsize>(sample_count));
        if (result == nullptr) return nullptr;
        env->SetFloatArrayRegion(result, 0, static_cast<jsize>(sample_count), audio);
        const int64_t output_copy_us = us_since(copy_start);
        g_last_timing_us = {lock_wait_us, input_prep_us, ort_run_us, output_copy_us, us_since(total_start), g_active_threads, g_active_xnnpack_threads};
        return result;
    } catch (const Ort::Exception& e) {
        LOGE("ONNX synthesis error: %s", e.what());
        throw_runtime(env, std::string("Vietnamese Kokoro synthesis failed: ") + e.what());
    } catch (const std::exception& e) {
        throw_runtime(env, std::string("Vietnamese Kokoro native synthesis failed: ") + e.what());
    }
    return nullptr;
}

extern "C" JNIEXPORT jlongArray JNICALL Java_com_CodeBySonu_VoxSherpa_vietnamese_VietnameseKokoroNative_getLastInferenceTimingMicros(JNIEnv* env, jobject) {
    std::lock_guard<std::mutex> lock(g_mutex);
    jlongArray out = env->NewLongArray(static_cast<jsize>(g_last_timing_us.size()));
    if (!out) return nullptr;
    std::array<jlong,7> values{};
    for (size_t i = 0; i < values.size(); ++i) values[i] = static_cast<jlong>(g_last_timing_us[i]);
    env->SetLongArrayRegion(out, 0, static_cast<jsize>(values.size()), values.data());
    return out;
}

extern "C" JNIEXPORT jstring JNICALL Java_com_CodeBySonu_VoxSherpa_vietnamese_VietnameseKokoroNative_benchmarkCpuThreads(
        JNIEnv* env, jobject, jstring model_path, jlongArray input_ids, jfloatArray ref_style,
        jfloat speed, jint warmup_runs, jint measured_runs) {
    const auto path = jstring_to_utf8(env, model_path);
    terminate_active_run();
    std::lock_guard<std::mutex> lock(g_mutex);
    try {
        const jsize id_count = env->GetArrayLength(input_ids);
        const jsize style_count = env->GetArrayLength(ref_style);
        if (id_count < 3 || id_count > 512 || style_count != 256) {
            throw_runtime(env, "Invalid CPU benchmark input."); return nullptr;
        }
        jlong* java_ids = env->GetLongArrayElements(input_ids, nullptr);
        jfloat* java_style = env->GetFloatArrayElements(ref_style, nullptr);
        std::vector<int64_t> ids(static_cast<size_t>(id_count));
        for (jsize i = 0; i < id_count; ++i) ids[static_cast<size_t>(i)] = static_cast<int64_t>(java_ids[i]);
        std::vector<float> style(java_style, java_style + style_count);
        env->ReleaseLongArrayElements(input_ids, java_ids, JNI_ABORT);
        env->ReleaseFloatArrayElements(ref_style, java_style, JNI_ABORT);

        const int warmups = std::max(1, static_cast<int>(warmup_runs));
        const int measures = std::max(2, static_cast<int>(measured_runs));
        const unsigned int detected_cpus = std::thread::hardware_concurrency();
        const unsigned int online_cpus = detected_cpus == 0 ? 4u : detected_cpus;
        std::vector<int> candidates{0};
        for (int candidate : {3, 4, 6}) {
            if (candidate <= static_cast<int>(online_cpus)) candidates.push_back(candidate);
        }
        if (candidates.size() == 1) candidates.push_back(std::max(1, std::min(4, static_cast<int>(online_cpus))));

        const size_t short_tokens = std::max<size_t>(12, std::min<size_t>(32, ids.size()));
        const size_t medium_tokens = std::max<size_t>(36, std::min<size_t>(64, ids.size() * 2));
        const std::array<std::vector<int64_t>,2> workloads{
                resize_ids(ids, short_tokens),
                resize_ids(ids, medium_tokens)
        };

        std::vector<CandidateBenchmark> results;
        results.reserve(candidates.size());
        int raw_best_threads = 0;
        int64_t raw_best_score = std::numeric_limits<int64_t>::max();

        g_session.reset();
        g_active_nnapi.store(false, std::memory_order_release);
        g_active_xnnpack.store(false, std::memory_order_release);
        g_active_xnnpack_threads = 0;
        for (int candidate : candidates) {
            CandidateBenchmark result;
            result.threads = candidate;
            auto load_start = Clock::now();
            auto session = create_session(path, candidate, false, false);
            result.load_ms = us_since(load_start) / 1000;

            for (size_t workload = 0; workload < workloads.size(); ++workload) {
                for (int i = 0; i < warmups; ++i) {
                    run_once(*session, workloads[workload], style, speed);
                }
                std::vector<int64_t> samples;
                samples.reserve(static_cast<size_t>(measures));
                for (int i = 0; i < measures; ++i) {
                    samples.push_back(run_once(*session, workloads[workload], style, speed).elapsed_us);
                }
                result.median_us[workload] = median_us(samples);
            }

            // TalkBack overwhelmingly favors short utterances; medium text prevents overfitting.
            result.score_us = result.median_us[0] * 2 + result.median_us[1];
            if (result.score_us < raw_best_score) {
                raw_best_score = result.score_us;
                raw_best_threads = candidate;
            }
            results.push_back(result);
            session.reset();
        }

        int best_threads = raw_best_threads;
        bool preferred_ort_default_for_affinity = false;
        if (!results.empty() && raw_best_threads != 0) {
            const int64_t default_score = results.front().score_us;
            if (default_score > 0
                    && static_cast<double>(default_score) <= static_cast<double>(raw_best_score) * 1.03) {
                best_threads = 0;
                preferred_ort_default_for_affinity = true;
            }
        }

        g_session = create_session(path, best_threads, false, false);
        g_active_threads = best_threads;
        g_active_nnapi.store(false, std::memory_order_release);

        std::ostringstream json;
        json << "{\"bestThreads\":" << best_threads
             << ",\"rawBestThreads\":" << raw_best_threads
             << ",\"preferredOrtDefaultForAffinity\":"
             << (preferred_ort_default_for_affinity ? "true" : "false")
             << ",\"onlineCpus\":" << online_cpus
             << ",\"workloadTokenCounts\":["
             << workloads[0].size() << "," << workloads[1].size() << "]"
             << ",\"candidates\":[";
        for (size_t i = 0; i < results.size(); ++i) {
            if (i != 0) json << ",";
            const CandidateBenchmark& r = results[i];
            json << "{\"threads\":" << r.threads
                 << ",\"autoAffinity\":" << (r.threads == 0 ? "true" : "false")
                 << ",\"loadMs\":" << r.load_ms
                 << ",\"shortMedianUs\":" << r.median_us[0]
                 << ",\"mediumMedianUs\":" << r.median_us[1]
                 << ",\"scoreUs\":" << r.score_us << "}";
        }
        json << "]}";
        const std::string out = json.str();
        LOGI("CPU benchmark %s", out.c_str());
        return env->NewStringUTF(out.c_str());
    } catch (const Ort::Exception& e) {
        throw_runtime(env, std::string("CPU benchmark failed: ") + e.what());
    } catch (const std::exception& e) {
        throw_runtime(env, std::string("CPU benchmark failed: ") + e.what());
    }
    return nullptr;
}
