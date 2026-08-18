#include <jni.h>
#include <android/log.h>
#include <onnxruntime_cxx_api.h>
#include <algorithm>
#include <array>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#define LOG_TAG "VoxSherpaKokoroVI"
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)

extern "C" {
void* sea_g2p_create(const char* dictionary_path);
char* sea_g2p_phonemize(void* handle, const char* text);
void sea_g2p_free_string(char* value);
void sea_g2p_destroy(void* handle);
}

namespace {
Ort::Env g_env(ORT_LOGGING_LEVEL_WARNING, "VoxSherpaKokoroVietnamese");
std::unique_ptr<Ort::Session> g_session;
std::mutex g_mutex;

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
extern "C" JNIEXPORT jboolean JNICALL Java_com_CodeBySonu_VoxSherpa_vietnamese_VietnameseKokoroNative_createEngine(JNIEnv* env, jobject, jstring model_path) {
    const auto path = jstring_to_utf8(env, model_path);
    try {
        std::lock_guard<std::mutex> lock(g_mutex);
        Ort::SessionOptions options;
        const unsigned int hardware_threads = std::thread::hardware_concurrency();
        const int inference_threads = static_cast<int>(std::min(6u, std::max(2u, hardware_threads)));
        options.SetIntraOpNumThreads(inference_threads);
        options.SetInterOpNumThreads(1);
        options.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
        options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        g_session = std::make_unique<Ort::Session>(g_env, path.c_str(), options);
        LOGI("Vietnamese Kokoro loaded with %d inference threads", inference_threads);
        return JNI_TRUE;
    } catch (const Ort::Exception& e) {
        LOGE("ONNX load error: %s", e.what());
        throw_runtime(env, std::string("Vietnamese Kokoro model load failed: ") + e.what());
    } catch (const std::exception& e) {
        throw_runtime(env, std::string("Vietnamese Kokoro native load failed: ") + e.what());
    }
    return JNI_FALSE;
}
extern "C" JNIEXPORT void JNICALL Java_com_CodeBySonu_VoxSherpa_vietnamese_VietnameseKokoroNative_destroyEngine(JNIEnv*, jobject) {
    std::lock_guard<std::mutex> lock(g_mutex);
    g_session.reset();
}
extern "C" JNIEXPORT jfloatArray JNICALL Java_com_CodeBySonu_VoxSherpa_vietnamese_VietnameseKokoroNative_synthesize(JNIEnv* env, jobject, jlongArray input_ids, jfloatArray ref_style, jfloat speed) {
    try {
        std::lock_guard<std::mutex> lock(g_mutex);
        if (!g_session) { throw_runtime(env, "Vietnamese Kokoro model is not loaded."); return nullptr; }
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
        auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        const std::array<int64_t,2> ids_shape{1, static_cast<int64_t>(id_count)};
        const std::array<int64_t,2> style_shape{1, 256};
        const std::array<int64_t,0> scalar_shape{};
        auto ids_tensor = Ort::Value::CreateTensor<int64_t>(memory, ids.data(), ids.size(), ids_shape.data(), ids_shape.size());
        auto style_tensor = Ort::Value::CreateTensor<float>(memory, style.data(), style.size(), style_shape.data(), style_shape.size());
        auto speed_tensor = Ort::Value::CreateTensor<float>(memory, &speed, 1, scalar_shape.data(), scalar_shape.size());
        const char* input_names[] = {"input_ids", "ref_s", "speed"};
        const char* output_names[] = {"waveform", "duration"};
        std::array<Ort::Value,3> inputs{std::move(ids_tensor), std::move(style_tensor), std::move(speed_tensor)};
        auto outputs = g_session->Run(Ort::RunOptions{nullptr}, input_names, inputs.data(), inputs.size(), output_names, 2);
        const float* audio = outputs[0].GetTensorData<float>();
        const size_t sample_count = outputs[0].GetTensorTypeAndShapeInfo().GetElementCount();
        if (sample_count == 0 || sample_count > 12000000) { throw_runtime(env, "Vietnamese Kokoro returned invalid audio."); return nullptr; }
        jfloatArray result = env->NewFloatArray(static_cast<jsize>(sample_count));
        if (result == nullptr) return nullptr;
        env->SetFloatArrayRegion(result, 0, static_cast<jsize>(sample_count), audio);
        return result;
    } catch (const Ort::Exception& e) {
        LOGE("ONNX synthesis error: %s", e.what());
        throw_runtime(env, std::string("Vietnamese Kokoro synthesis failed: ") + e.what());
    } catch (const std::exception& e) {
        throw_runtime(env, std::string("Vietnamese Kokoro native synthesis failed: ") + e.what());
    }
    return nullptr;
}
