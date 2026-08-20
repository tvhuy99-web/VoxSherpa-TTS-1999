from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CPP = ROOT / "app/src/main/cpp/kokoro_vi/kokoro_vi_jni.cpp"
ENGINE = ROOT / "app/src/main/java/com/CodeBySonu/VoxSherpa/vietnamese/VietnameseKokoroEngine.java"
ACTIVITY = ROOT / "app/src/main/java/com/CodeBySonu/VoxSherpa/system/TtsDiagnosticsActivity.java"


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{label}: expected exactly one match, found {count}")
    return text.replace(old, new, 1)


# Native benchmark: keep the normal production CPU session creation path untouched.
# Only the benchmark function creates experimental SessionOptions candidates.
cpp = CPP.read_text(encoding="utf-8")
marker = 'extern "C" JNIEXPORT jstring JNICALL Java_com_CodeBySonu_VoxSherpa_vietnamese_VietnameseKokoroNative_benchmarkCpuThreads('
start = cpp.find(marker)
if start < 0:
    raise SystemExit("Native CPU benchmark function marker not found")

new_benchmark = r'''extern "C" JNIEXPORT jstring JNICALL Java_com_CodeBySonu_VoxSherpa_vietnamese_VietnameseKokoroNative_benchmarkCpuThreads(
        JNIEnv* env, jobject, jstring model_path, jlongArray input_ids, jfloatArray ref_style,
        jfloat speed, jint warmup_runs, jint measured_runs) {
    const auto path = jstring_to_utf8(env, model_path);
    terminate_active_run();
    std::lock_guard<std::mutex> lock(g_mutex);
    try {
        const jsize id_count = env->GetArrayLength(input_ids);
        const jsize style_count = env->GetArrayLength(ref_style);
        if (id_count < 3 || id_count > 512 || style_count != 256) {
            throw_runtime(env, "Invalid CPU runtime benchmark input.");
            return nullptr;
        }

        jlong* java_ids = env->GetLongArrayElements(input_ids, nullptr);
        jfloat* java_style = env->GetFloatArrayElements(ref_style, nullptr);
        if (java_ids == nullptr || java_style == nullptr) {
            if (java_ids != nullptr) env->ReleaseLongArrayElements(input_ids, java_ids, JNI_ABORT);
            if (java_style != nullptr) env->ReleaseFloatArrayElements(ref_style, java_style, JNI_ABORT);
            throw_runtime(env, "Not enough memory for CPU runtime benchmark input.");
            return nullptr;
        }

        std::vector<int64_t> ids(static_cast<size_t>(id_count));
        for (jsize i = 0; i < id_count; ++i) ids[static_cast<size_t>(i)] = static_cast<int64_t>(java_ids[i]);
        std::vector<float> style(java_style, java_style + style_count);
        env->ReleaseLongArrayElements(input_ids, java_ids, JNI_ABORT);
        env->ReleaseFloatArrayElements(ref_style, java_style, JNI_ABORT);

        const int warmups = std::max(1, static_cast<int>(warmup_runs));
        const int measures = std::max(2, static_cast<int>(measured_runs));
        const unsigned int detected_cpus = std::thread::hardware_concurrency();
        const unsigned int online_cpus = detected_cpus == 0 ? 4u : detected_cpus;

        struct RuntimeConfig {
            std::string name;
            int threads;
            int dynamic_block_base;
            bool parallel;
            int inter_threads;
        };
        struct RuntimeResult {
            RuntimeConfig config;
            int64_t load_ms = 0;
            std::array<int64_t,4> median_us{0,0,0,0};
            int64_t weighted_score_us = 0;
        };

        auto create_runtime_session = [&](const RuntimeConfig& config) -> std::unique_ptr<Ort::Session> {
            Ort::SessionOptions options;
            if (config.threads > 0) options.SetIntraOpNumThreads(config.threads);
            options.SetInterOpNumThreads(config.parallel ? std::max(1, config.inter_threads) : 1);
            options.SetExecutionMode(config.parallel ? ExecutionMode::ORT_PARALLEL : ExecutionMode::ORT_SEQUENTIAL);
            options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
            if (config.dynamic_block_base > 0) {
                const std::string value = std::to_string(config.dynamic_block_base);
                options.AddConfigEntry("session.dynamic_block_base", value.c_str());
            }
            return std::make_unique<Ort::Session>(g_env, path.c_str(), options);
        };

        const std::array<size_t,4> token_counts{21, 50, 71, 110};
        const std::array<int64_t,4> weights{4, 3, 2, 1};
        const std::array<std::vector<int64_t>,4> workloads{
                resize_ids(ids, token_counts[0]),
                resize_ids(ids, token_counts[1]),
                resize_ids(ids, token_counts[2]),
                resize_ids(ids, token_counts[3])
        };

        auto benchmark_config = [&](const RuntimeConfig& config) -> RuntimeResult {
            RuntimeResult result;
            result.config = config;
            auto load_start = Clock::now();
            auto session = create_runtime_session(config);
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
                result.weighted_score_us += result.median_us[workload] * weights[workload];
            }
            return result;
        };

        const RuntimeConfig baseline{"baseline", 0, 0, false, 1};
        RuntimeResult cpu_before = benchmark_config(baseline);

        std::vector<RuntimeConfig> candidates{
                {"dynamic2_default", 0, 2, false, 1},
                {"dynamic4_default", 0, 4, false, 1},
                {"dynamic8_default", 0, 8, false, 1},
                {"dynamic4_threads3", 3, 4, false, 1},
                {"dynamic4_threads4", 4, 4, false, 1},
                {"dynamic4_threads6", 6, 4, false, 1},
                {"parallel2_dynamic4", 0, 4, true, 2}
        };

        std::vector<RuntimeResult> results;
        for (const RuntimeConfig& candidate : candidates) {
            if (candidate.threads > static_cast<int>(online_cpus)) continue;
            results.push_back(benchmark_config(candidate));
        }

        RuntimeResult cpu_after = benchmark_config(baseline);
        RuntimeResult cpu_reference;
        cpu_reference.config = baseline;
        cpu_reference.load_ms = (cpu_before.load_ms + cpu_after.load_ms) / 2;
        for (size_t i = 0; i < cpu_reference.median_us.size(); ++i) {
            cpu_reference.median_us[i] = (cpu_before.median_us[i] + cpu_after.median_us[i]) / 2;
            cpu_reference.weighted_score_us += cpu_reference.median_us[i] * weights[i];
        }

        const RuntimeResult* raw_best = nullptr;
        for (const RuntimeResult& result : results) {
            if (raw_best == nullptr || result.weighted_score_us < raw_best->weighted_score_us) {
                raw_best = &result;
            }
        }

        RuntimeConfig selected = baseline;
        double best_vs_cpu_ratio = 1.0;
        bool material_gain = false;
        if (raw_best != nullptr && cpu_reference.weighted_score_us > 0) {
            best_vs_cpu_ratio = static_cast<double>(raw_best->weighted_score_us)
                    / static_cast<double>(cpu_reference.weighted_score_us);
            // Keep the simple production baseline unless a tuning candidate wins by at least 2%.
            material_gain = best_vs_cpu_ratio <= 0.98;
            if (material_gain) selected = raw_best->config;
        }

        g_session.reset();
        g_session = create_runtime_session(selected);
        g_active_threads = selected.threads;
        g_active_nnapi.store(false, std::memory_order_release);

        auto append_result = [](std::ostringstream& json, const RuntimeResult& result) {
            json << "{\"mode\":\"" << result.config.name << "\""
                 << ",\"threads\":" << result.config.threads
                 << ",\"dynamicBlockBase\":" << result.config.dynamic_block_base
                 << ",\"parallel\":" << (result.config.parallel ? "true" : "false")
                 << ",\"interThreads\":" << result.config.inter_threads
                 << ",\"loadMs\":" << result.load_ms
                 << ",\"mediansUs\":["
                 << result.median_us[0] << "," << result.median_us[1] << ","
                 << result.median_us[2] << "," << result.median_us[3] << "]"
                 << ",\"weightedScoreUs\":" << result.weighted_score_us << "}";
        };

        std::ostringstream json;
        json << "{\"bestThreads\":" << selected.threads
             << ",\"selectedMode\":\"" << selected.name << "\""
             << ",\"selectedThreads\":" << selected.threads
             << ",\"selectedDynamicBlockBase\":" << selected.dynamic_block_base
             << ",\"selectedParallel\":" << (selected.parallel ? "true" : "false")
             << ",\"selectedInterThreads\":" << selected.inter_threads
             << ",\"bestVsCpuRatio\":" << best_vs_cpu_ratio
             << ",\"materialGain\":" << (material_gain ? "true" : "false")
             << ",\"onlineCpus\":" << online_cpus
             << ",\"warmupRuns\":" << warmups
             << ",\"measuredRuns\":" << measures
             << ",\"workloadTokenCounts\":[21,50,71,110]"
             << ",\"weights\":[4,3,2,1]"
             << ",\"cpuBefore\":";
        append_result(json, cpu_before);
        json << ",\"cpuAfter\":";
        append_result(json, cpu_after);
        json << ",\"cpuReference\":";
        append_result(json, cpu_reference);
        json << ",\"candidates\":[";
        for (size_t i = 0; i < results.size(); ++i) {
            if (i != 0) json << ",";
            append_result(json, results[i]);
        }
        json << "]}";

        const std::string out = json.str();
        LOGI("CPU runtime tuning benchmark %s", out.c_str());
        return env->NewStringUTF(out.c_str());
    } catch (const Ort::Exception& e) {
        throw_runtime(env, std::string("CPU runtime tuning benchmark failed: ") + e.what());
    } catch (const std::exception& e) {
        throw_runtime(env, std::string("CPU runtime tuning benchmark failed: ") + e.what());
    }
    return nullptr;
}
'''
CPP.write_text(cpp[:start] + new_benchmark, encoding="utf-8")

# Java engine: do not persist an experimental scheduler/thread choice yet.
engine = ENGINE.read_text(encoding="utf-8")
method_start = engine.find("    public synchronized String benchmarkCpuThreads(Context context) throws Exception {")
method_end = engine.find("    private synchronized void prepare(Context context) throws Exception {", method_start)
if method_start < 0 or method_end < 0:
    raise SystemExit("Engine CPU benchmark method markers not found")

new_engine_method = r'''    public synchronized String benchmarkCpuThreads(Context context) throws Exception {
        Context app = context.getApplicationContext();
        appContext = app;
        long totalStart = System.nanoTime();
        SharedPreferences prefs = app.getSharedPreferences(PERF_PREFS, Context.MODE_PRIVATE);
        // This experiment is CPU-only. Keep the production preference at ORT default so
        // a process restart always returns to the known-safe baseline until we approve a winner.
        prefs.edit().putBoolean(PREF_NNAPI_ENABLED, false).putInt(PREF_CPU_THREADS, 0).apply();
        TtsDiagnostics.info(app, "benchmark", "cpu_runtime_start",
                "CPU-only runtime tuning: baseline vs dynamic block 2/4/8, dynamic4 with 3/4/6 threads, "
                        + "and parallel2+dynamic4. Workloads=21/50/71/110 tokens, weights=4/3/2/1, "
                        + "1 warmup + 2 measured runs; baseline is sampled before and after tuning candidates.");
        prepare(app);
        String phonemes = phonemize(WARMUP_TEXT);
        long[] ids = tokenIds(phonemes);
        float[] style = selectStyle(VietnameseKokoroVoice.DEFAULT, phonemes.length());
        String json;
        try {
            json = nativeBridge.benchmarkCpuThreads(assets.model.getAbsolutePath(), ids, style, 1.0f, 1, 2);
        } catch (Exception e) {
            modelReady = false;
            modelWarm = false;
            activeNnapi = false;
            throw e;
        }

        JSONObject result = new JSONObject(json);
        int selectedThreads = result.optInt("selectedThreads", 0);
        String selectedMode = result.optString("selectedMode", "baseline");
        double ratio = result.optDouble("bestVsCpuRatio", 1.0);
        boolean materialGain = result.optBoolean("materialGain", false);
        activeCpuThreads = selectedThreads;
        modelReady = true;
        modelWarm = false;
        activeNnapi = false;

        if (!warmCurrentSession(app, "cpu_runtime_tuning_selected:" + selectedMode)) {
            throw new IllegalStateException("Vietnamese Kokoro selected CPU runtime warm-up was interrupted.");
        }

        TtsDiagnostics.info(app, "benchmark", "cpu_runtime_complete",
                "elapsedMs=" + elapsedMs(totalStart) + ", selectedMode=" + selectedMode
                        + ", selectedThreads=" + selectedThreads + ", bestVsCpuRatio=" + ratio
                        + ", materialGain=" + materialGain + ", result=" + json
                        + ", note=selection is active only for this process; restart returns to ORT default baseline");
        return json;
    }

'''
ENGINE.write_text(engine[:method_start] + new_engine_method + engine[method_end:], encoding="utf-8")

# Accessible diagnostics wording for the CPU-only experiment.
activity = ACTIVITY.read_text(encoding="utf-8")
activity = replace_once(
    activity,
    'benchmark.setText("CPU benchmark: default / 3 / 4 / 5 / 6");',
    'benchmark.setText("CPU runtime tuning benchmark");',
    "initial benchmark button text",
)
activity = replace_once(
    activity,
    '"Run a deep CPU benchmark for ONNX Runtime default, three, four, five, and six threads. "\n                        + "Each mode uses two warmup runs and five measured runs. Speech may be temporarily blocked while the model is tested.");',
    '"Run a CPU-only ONNX Runtime tuning benchmark. It compares the current baseline with dynamic block scheduling, "\n                        + "several CPU thread counts, and one light parallel mode across 21, 50, 71, and 110 token workloads. "\n                        + "The benchmark keeps an experimental winner only when it is at least two percent faster. Speech is blocked while testing.");',
    "benchmark content description",
)
activity = replace_once(
    activity,
    'benchmark.setText("CPU benchmark running…");',
    'benchmark.setText("CPU runtime tuning running…");',
    "running benchmark button text",
)
activity = replace_once(
    activity,
    'TtsDiagnostics.info(this, "benchmark", "ui_requested",\n                "User started lightweight ORT default/3/4/6 CPU benchmark across short/medium TalkBack workloads.");',
    'TtsDiagnostics.info(this, "benchmark", "cpu_runtime_ui_requested",\n                "User started CPU-only runtime tuning benchmark across 21/50/71/110 token TalkBack workloads.");',
    "benchmark UI log",
)
activity = activity.replace(
    'benchmark.setText("CPU benchmark: default / 3 / 4 / 5 / 6");',
    'benchmark.setText("CPU runtime tuning benchmark");',
)
activity = replace_once(
    activity,
    'Toast.makeText(this, "CPU benchmark complete; fastest CPU mode saved.", Toast.LENGTH_LONG).show();',
    'Toast.makeText(this, "CPU runtime tuning complete. The best session is active for this process; share the log.", Toast.LENGTH_LONG).show();',
    "benchmark completion toast",
)
activity = replace_once(
    activity,
    'Toast.makeText(this, "CPU benchmark failed. See TTS Logs.", Toast.LENGTH_LONG).show();',
    'Toast.makeText(this, "CPU runtime tuning failed. See TTS Logs.", Toast.LENGTH_LONG).show();',
    "benchmark failure toast",
)
ACTIVITY.write_text(activity, encoding="utf-8")

print("CPU runtime tuning experiment patch applied successfully")
