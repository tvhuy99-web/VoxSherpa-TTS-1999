from pathlib import Path


def replace_once(text, old, new, label):
    if old not in text:
        raise SystemExit(f"Missing patch anchor: {label}")
    return text.replace(old, new, 1)


p = Path("app/src/main/java/com/CodeBySonu/VoxSherpa/vietnamese/VietnameseKokoroNative.java")
s = p.read_text()
s = replace_once(s,
    '    public native boolean createEngine(String modelPath, int cpuThreads, boolean useNnapi, boolean useXnnpack);\n',
    '    public native boolean createEngine(String modelPath, int cpuThreads, boolean useNnapi, boolean useXnnpack, int xnnpackThreads);\n',
    "native createEngine xnnpack threads")
s = replace_once(s,
'''    public native String benchmarkCpuThreads(
            String modelPath,
            long[] inputIds,
            float[] refStyle,
            float speed,
            int warmupRuns,
            int measuredRuns
    );
''',
'''    public native String benchmarkCpuThreads(
            String modelPath,
            long[] inputIds,
            float[] refStyle,
            float speed,
            int warmupRuns,
            int measuredRuns
    );

    /** Controlled CPU-default vs XNNPACK 1/2/4/6/8 thread benchmark. */
    public native String benchmarkXnnpackThreads(
            String modelPath,
            long[] inputIds,
            float[] refStyle,
            float speed,
            int warmupRuns,
            int measuredRuns
    );
''', "native benchmark declaration")
p.write_text(s)

p = Path("app/src/main/java/com/CodeBySonu/VoxSherpa/vietnamese/VietnameseKokoroEngine.java")
s = p.read_text()
s = replace_once(s,
    '    private static final String PREF_RUNTIME_BACKEND = "runtime_backend";\n',
    '    private static final String PREF_RUNTIME_BACKEND = "runtime_backend";\n    private static final String PREF_XNNPACK_THREADS = "xnnpack_threads";\n',
    "engine xnnpack pref")
s = replace_once(s,
    '    private volatile boolean activeXnnpack;\n    private volatile int activeCpuThreads = 0;\n',
    '    private volatile boolean activeXnnpack;\n    private volatile int activeCpuThreads = 0;\n    private volatile int activeXnnpackThreads = 0;\n',
    "engine active xnnpack thread field")
s = replace_once(s,
'''    public boolean isXnnpackRequested(Context context) {
        return BuildConfig.KOKORO_XNNPACK_AB
                && BACKEND_XNNPACK.equals(requestedRuntimeBackend(context));
    }

''',
'''    public boolean isXnnpackRequested(Context context) {
        return BuildConfig.KOKORO_XNNPACK_AB
                && BACKEND_XNNPACK.equals(requestedRuntimeBackend(context));
    }

    public int selectedXnnpackThreads(Context context) {
        if (context == null) return 8;
        int saved = context.getSharedPreferences(PERF_PREFS, Context.MODE_PRIVATE)
                .getInt(PREF_XNNPACK_THREADS, 8);
        return Math.max(1, Math.min(8, saved));
    }

''', "engine selected xnnpack threads")
s = replace_once(s,
    '                "requestedBackend=" + requested + ", CPU mode uses ORT default; XNNPACK uses its dedicated threadpool");\n',
    '                "requestedBackend=" + requested + ", CPU mode uses ORT default; XNNPACK threads="\n                        + selectedXnnpackThreads(app));\n',
    "backend switch log")
s = replace_once(s,
'''        activeNnapi = false;
        activeXnnpack = false;
        activeCpuThreads = 0;

        prepare(app);
''',
'''        activeNnapi = false;
        activeXnnpack = false;
        activeCpuThreads = 0;
        activeXnnpackThreads = 0;

        prepare(app);
''', "backend switch reset")
s = replace_once(s,
'''        int saved = prefs.getInt(PREF_CPU_THREADS, 0);
        boolean nnapiRequested = prefs.getBoolean(PREF_NNAPI_ENABLED, false);
        synchronized (this) {
''',
'''        int saved = prefs.getInt(PREF_CPU_THREADS, 0);
        boolean nnapiRequested = prefs.getBoolean(PREF_NNAPI_ENABLED, false);
        int savedXnnpackThreads = selectedXnnpackThreads(context);
        synchronized (this) {
''', "performance selected threads")
s = replace_once(s,
'''                    + ", nnapiRequested=" + nnapiRequested + ", activeCpuThreads=" + activeCpuThreads
                    + ", savedCpuThreads=" + saved + " (0=ORT default/auto-affinity)"
''',
'''                    + ", nnapiRequested=" + nnapiRequested + ", activeCpuThreads=" + activeCpuThreads
                    + ", activeXnnpackThreads=" + activeXnnpackThreads
                    + ", savedXnnpackThreads=" + savedXnnpackThreads
                    + ", savedCpuThreads=" + saved + " (0=ORT default/auto-affinity)"
''', "performance xnnpack threads")

benchmark_method = r'''    public synchronized String benchmarkXnnpackThreads(Context context) throws Exception {
        if (!BuildConfig.KOKORO_XNNPACK_AB) {
            throw new IllegalStateException("XNNPACK benchmark is only available in the XNNPACK experiment build.");
        }
        Context app = context.getApplicationContext();
        appContext = app;
        long totalStart = System.nanoTime();
        SharedPreferences prefs = app.getSharedPreferences(PERF_PREFS, Context.MODE_PRIVATE);
        TtsDiagnostics.info(app, "benchmark", "xnnpack_start",
                "Controlled benchmark: CPU default vs XNNPACK 1/2/4/6/8 threads; "
                        + "workloads=21/50/71/110 tokens, 1 warmup + 2 measured runs each. "
                        + "CPU is sampled before and after XNNPACK to reduce thermal-order bias; score weights=4/3/2/1.");
        prepare(app);
        String phonemes = phonemize(WARMUP_TEXT);
        long[] ids = tokenIds(phonemes);
        float[] style = selectStyle(VietnameseKokoroVoice.DEFAULT, phonemes.length());
        String json;
        try {
            json = nativeBridge.benchmarkXnnpackThreads(
                    assets.model.getAbsolutePath(), ids, style, 1.0f, 1, 2);
        } catch (Exception e) {
            try { nativeBridge.destroyEngine(); } catch (Throwable ignored) {}
            modelReady = false;
            modelWarm = false;
            activeNnapi = false;
            activeXnnpack = false;
            activeCpuThreads = 0;
            activeXnnpackThreads = 0;
            prefs.edit().putString(PREF_RUNTIME_BACKEND, BACKEND_CPU).apply();
            try {
                prepare(app);
                warmCurrentSession(app, "xnnpack_benchmark_failure_cpu_restore");
            } catch (Throwable restoreError) {
                TtsDiagnostics.error(app, "benchmark", "xnnpack_restore_failed",
                        restoreError.toString(), restoreError);
            }
            throw e;
        }
        JSONObject result = new JSONObject(json);
        int bestThreads = result.getInt("bestXnnpackThreads");
        String recommended = result.getString("recommendedBackend");
        double bestRatio = result.getDouble("bestXnnpackVsCpuRatio");
        if (!BACKEND_XNNPACK.equals(recommended)) recommended = BACKEND_CPU;
        prefs.edit()
                .putInt(PREF_XNNPACK_THREADS, bestThreads)
                .putString(PREF_RUNTIME_BACKEND, recommended)
                .putInt(PREF_CPU_THREADS, 0)
                .putBoolean(PREF_NNAPI_ENABLED, false)
                .apply();
        try { nativeBridge.destroyEngine(); } catch (Throwable ignored) {}
        modelReady = false;
        modelWarm = false;
        activeNnapi = false;
        activeXnnpack = false;
        activeCpuThreads = 0;
        activeXnnpackThreads = 0;
        prepare(app);
        if (!modelWarm && !warmCurrentSession(app, "xnnpack_benchmark_selected:" + recommended)) {
            throw new IllegalStateException("XNNPACK benchmark winner warm-up was interrupted.");
        }
        TtsDiagnostics.info(app, "benchmark", "xnnpack_complete",
                "elapsedMs=" + elapsedMs(totalStart)
                        + ", bestXnnpackThreads=" + bestThreads
                        + ", bestXnnpackVsCpuRatio=" + bestRatio
                        + ", recommendedBackend=" + recommended
                        + ", result=" + json
                        + ", providerAfterBenchmark=" + activeProviderLabel());
        return json;
    }

'''
s = replace_once(s,
    '    private synchronized void prepare(Context context) throws Exception {\n',
    benchmark_method + '    private synchronized void prepare(Context context) throws Exception {\n',
    "engine benchmark method")
s = replace_once(s,
'''            boolean requestedXnnpack = isXnnpackRequested(context);
            boolean requestedNnapi = !BuildConfig.KOKORO_XNNPACK_AB && !requestedXnnpack
                    && prefs.getBoolean(PREF_NNAPI_ENABLED, false);
            modelReady = nativeBridge.createEngine(assets.model.getAbsolutePath(), activeCpuThreads,
                    requestedNnapi, requestedXnnpack);
''',
'''            boolean requestedXnnpack = isXnnpackRequested(context);
            int requestedXnnpackThreads = selectedXnnpackThreads(context);
            boolean requestedNnapi = !BuildConfig.KOKORO_XNNPACK_AB && !requestedXnnpack
                    && prefs.getBoolean(PREF_NNAPI_ENABLED, false);
            modelReady = nativeBridge.createEngine(assets.model.getAbsolutePath(), activeCpuThreads,
                    requestedNnapi, requestedXnnpack, requestedXnnpackThreads);
''', "engine createEngine args")
s = replace_once(s,
'''            activeXnnpack = nativeBridge.isXnnpackActive();
            activeNnapi = nativeBridge.isNnapiActive();
''',
'''            activeXnnpack = nativeBridge.isXnnpackActive();
            activeNnapi = nativeBridge.isNnapiActive();
            activeXnnpackThreads = activeXnnpack ? requestedXnnpackThreads : 0;
''', "engine active xnnpack threads")
s = replace_once(s,
'''                            + ", xnnpackRequested=" + requestedXnnpack
                            + ", cpuThreads=" + threadLabel(activeCpuThreads));
''',
'''                            + ", xnnpackRequested=" + requestedXnnpack
                            + ", xnnpackThreads=" + activeXnnpackThreads
                            + ", cpuThreads=" + threadLabel(activeCpuThreads));
''', "model ready thread log")
s = replace_once(s,
'''        activeNnapi = false;
        activeXnnpack = false;
        assets = null;
''',
'''        activeNnapi = false;
        activeXnnpack = false;
        activeXnnpackThreads = 0;
        assets = null;
''', "release xnnpack threads")
p.write_text(s)

p = Path("app/src/main/cpp/kokoro_vi/kokoro_vi_jni.cpp")
s = p.read_text()
s = replace_once(s,
'''std::unique_ptr<Ort::Session> create_session(
        const std::string& path, int cpu_threads, bool use_nnapi, bool use_xnnpack) {
''',
'''std::unique_ptr<Ort::Session> create_session(
        const std::string& path, int cpu_threads, bool use_nnapi, bool use_xnnpack,
        int xnnpack_threads_override = 0) {
''', "native create_session override")
s = replace_once(s,
'''        const unsigned int detected = std::thread::hardware_concurrency();
        xnnpack_threads = std::max(1, std::min(8, static_cast<int>(detected == 0 ? 4u : detected)));
''',
'''        const unsigned int detected = std::thread::hardware_concurrency();
        const int auto_threads = std::max(1, std::min(8, static_cast<int>(detected == 0 ? 4u : detected)));
        xnnpack_threads = xnnpack_threads_override > 0
                ? std::max(1, std::min(8, xnnpack_threads_override))
                : auto_threads;
''', "native xnnpack selection")
s = replace_once(s,
'''struct CandidateBenchmark {
    int threads = 0;
    int64_t load_ms = 0;
    std::array<int64_t,3> median_us{0,0,0};
    int64_t score_us = 0;
};
''',
'''struct CandidateBenchmark {
    int threads = 0;
    int64_t load_ms = 0;
    std::array<int64_t,3> median_us{0,0,0};
    int64_t score_us = 0;
};

struct XnnpackCandidateBenchmark {
    int threads = 0;
    int64_t load_ms = 0;
    std::array<int64_t,4> median_us{0,0,0,0};
    int64_t weighted_score_us = 0;
};
''', "native benchmark result struct")
s = replace_once(s,
'''extern "C" JNIEXPORT jboolean JNICALL Java_com_CodeBySonu_VoxSherpa_vietnamese_VietnameseKokoroNative_createEngine(
        JNIEnv* env, jobject, jstring model_path, jint cpu_threads, jboolean use_nnapi, jboolean use_xnnpack) {
''',
'''extern "C" JNIEXPORT jboolean JNICALL Java_com_CodeBySonu_VoxSherpa_vietnamese_VietnameseKokoroNative_createEngine(
        JNIEnv* env, jobject, jstring model_path, jint cpu_threads, jboolean use_nnapi, jboolean use_xnnpack,
        jint xnnpack_threads) {
''', "native createEngine JNI signature")
s = replace_once(s,
'''                const unsigned int detected = std::thread::hardware_concurrency();
                const int xnn_threads = std::max(1, std::min(8, static_cast<int>(detected == 0 ? 4u : detected)));
                g_session = create_session(path, 1, false, true);
''',
'''                const unsigned int detected = std::thread::hardware_concurrency();
                const int auto_threads = std::max(1, std::min(8, static_cast<int>(detected == 0 ? 4u : detected)));
                const int xnn_threads = xnnpack_threads > 0
                        ? std::max(1, std::min(8, static_cast<int>(xnnpack_threads)))
                        : auto_threads;
                g_session = create_session(path, 1, false, true, xnn_threads);
''', "native production XNNPACK threads")

s += r'''

extern "C" JNIEXPORT jstring JNICALL Java_com_CodeBySonu_VoxSherpa_vietnamese_VietnameseKokoroNative_benchmarkXnnpackThreads(
        JNIEnv* env, jobject, jstring model_path, jlongArray input_ids, jfloatArray ref_style,
        jfloat speed, jint warmup_runs, jint measured_runs) {
    const auto path = jstring_to_utf8(env, model_path);
    terminate_active_run();
    std::lock_guard<std::mutex> lock(g_mutex);
    try {
        const jsize id_count = env->GetArrayLength(input_ids);
        const jsize style_count = env->GetArrayLength(ref_style);
        if (id_count < 3 || id_count > 512 || style_count != 256) {
            throw_runtime(env, "Invalid XNNPACK benchmark input."); return nullptr;
        }
        jlong* java_ids = env->GetLongArrayElements(input_ids, nullptr);
        jfloat* java_style = env->GetFloatArrayElements(ref_style, nullptr);
        if (java_ids == nullptr || java_style == nullptr) {
            if (java_ids) env->ReleaseLongArrayElements(input_ids, java_ids, JNI_ABORT);
            if (java_style) env->ReleaseFloatArrayElements(ref_style, java_style, JNI_ABORT);
            throw_runtime(env, "Not enough memory for XNNPACK benchmark input."); return nullptr;
        }
        std::vector<int64_t> ids(static_cast<size_t>(id_count));
        for (jsize i = 0; i < id_count; ++i) ids[static_cast<size_t>(i)] = static_cast<int64_t>(java_ids[i]);
        std::vector<float> style(java_style, java_style + style_count);
        env->ReleaseLongArrayElements(input_ids, java_ids, JNI_ABORT);
        env->ReleaseFloatArrayElements(ref_style, java_style, JNI_ABORT);
        const int warmups = std::max(1, static_cast<int>(warmup_runs));
        const int measures = std::max(2, static_cast<int>(measured_runs));
        const std::array<std::vector<int64_t>,4> workloads{
                resize_ids(ids, 21), resize_ids(ids, 50), resize_ids(ids, 71), resize_ids(ids, 110)};
        const std::array<int64_t,4> weights{4, 3, 2, 1};
        auto measure = [&](Ort::Session& session, int threads, int64_t load_ms) {
            XnnpackCandidateBenchmark result;
            result.threads = threads;
            result.load_ms = load_ms;
            for (size_t w = 0; w < workloads.size(); ++w) {
                for (int i = 0; i < warmups; ++i) run_once(session, workloads[w], style, speed);
                std::vector<int64_t> samples;
                samples.reserve(static_cast<size_t>(measures));
                for (int i = 0; i < measures; ++i)
                    samples.push_back(run_once(session, workloads[w], style, speed).elapsed_us);
                result.median_us[w] = median_us(samples);
                result.weighted_score_us += result.median_us[w] * weights[w];
            }
            return result;
        };
        g_session.reset();
        g_active_nnapi.store(false, std::memory_order_release);
        g_active_xnnpack.store(false, std::memory_order_release);
        g_active_xnnpack_threads = 0;
        auto cpu_before_load_start = Clock::now();
        auto cpu_before_session = create_session(path, 0, false, false);
        XnnpackCandidateBenchmark cpu_before = measure(*cpu_before_session, 0, us_since(cpu_before_load_start) / 1000);
        cpu_before_session.reset();
        const unsigned int detected = std::thread::hardware_concurrency();
        const int online_cpus = std::max(1, static_cast<int>(detected == 0 ? 4u : detected));
        std::vector<int> candidates;
        for (int t : {1, 8, 2, 6, 4}) if (t <= online_cpus) candidates.push_back(t);
        if (candidates.empty()) candidates.push_back(1);
        std::vector<XnnpackCandidateBenchmark> results;
        results.reserve(candidates.size());
        int best_threads = candidates.front();
        int64_t best_score = std::numeric_limits<int64_t>::max();
        for (int threads : candidates) {
            auto load_start = Clock::now();
            auto session = create_session(path, 1, false, true, threads);
            XnnpackCandidateBenchmark result = measure(*session, threads, us_since(load_start) / 1000);
            if (result.weighted_score_us < best_score) { best_score = result.weighted_score_us; best_threads = threads; }
            results.push_back(result);
            session.reset();
        }
        auto cpu_after_load_start = Clock::now();
        auto cpu_after_session = create_session(path, 0, false, false);
        XnnpackCandidateBenchmark cpu_after = measure(*cpu_after_session, 0, us_since(cpu_after_load_start) / 1000);
        cpu_after_session.reset();
        XnnpackCandidateBenchmark cpu;
        cpu.threads = 0;
        cpu.load_ms = (cpu_before.load_ms + cpu_after.load_ms) / 2;
        for (size_t w = 0; w < cpu.median_us.size(); ++w) {
            cpu.median_us[w] = (cpu_before.median_us[w] + cpu_after.median_us[w]) / 2;
            cpu.weighted_score_us += cpu.median_us[w] * weights[w];
        }
        const double ratio = cpu.weighted_score_us > 0
                ? static_cast<double>(best_score) / static_cast<double>(cpu.weighted_score_us) : 999.0;
        const bool xnnpack_wins_materially = ratio <= 0.95;
        std::ostringstream json;
        json << "{\"onlineCpus\":" << online_cpus
             << ",\"warmupRuns\":" << warmups << ",\"measuredRuns\":" << measures
             << ",\"workloadTokenCounts\":[" << workloads[0].size() << "," << workloads[1].size() << ","
             << workloads[2].size() << "," << workloads[3].size() << "]"
             << ",\"weights\":[4,3,2,1]"
             << ",\"cpuBefore\":{\"loadMs\":" << cpu_before.load_ms << ",\"mediansUs\":["
             << cpu_before.median_us[0] << "," << cpu_before.median_us[1] << "," << cpu_before.median_us[2] << "," << cpu_before.median_us[3] << "]}"
             << ",\"cpuAfter\":{\"loadMs\":" << cpu_after.load_ms << ",\"mediansUs\":["
             << cpu_after.median_us[0] << "," << cpu_after.median_us[1] << "," << cpu_after.median_us[2] << "," << cpu_after.median_us[3] << "]}"
             << ",\"cpu\":{\"loadMs\":" << cpu.load_ms << ",\"mediansUs\":["
             << cpu.median_us[0] << "," << cpu.median_us[1] << "," << cpu.median_us[2] << "," << cpu.median_us[3]
             << "],\"weightedScoreUs\":" << cpu.weighted_score_us << "}"
             << ",\"bestXnnpackThreads\":" << best_threads
             << ",\"bestXnnpackWeightedScoreUs\":" << best_score
             << ",\"bestXnnpackVsCpuRatio\":" << ratio
             << ",\"recommendedBackend\":\"" << (xnnpack_wins_materially ? "XNNPACK" : "CPU") << "\""
             << ",\"candidates\":[";
        for (size_t i = 0; i < results.size(); ++i) {
            if (i != 0) json << ",";
            const auto& r = results[i];
            json << "{\"threads\":" << r.threads << ",\"loadMs\":" << r.load_ms << ",\"mediansUs\":["
                 << r.median_us[0] << "," << r.median_us[1] << "," << r.median_us[2] << "," << r.median_us[3]
                 << "],\"weightedScoreUs\":" << r.weighted_score_us << ",\"vsCpuRatio\":"
                 << (cpu.weighted_score_us > 0 ? static_cast<double>(r.weighted_score_us) / static_cast<double>(cpu.weighted_score_us) : 999.0) << "}";
        }
        json << "]}";
        const std::string out = json.str();
        LOGI("XNNPACK thread benchmark %s", out.c_str());
        g_session.reset();
        g_active_xnnpack.store(false, std::memory_order_release);
        g_active_nnapi.store(false, std::memory_order_release);
        g_active_xnnpack_threads = 0;
        return env->NewStringUTF(out.c_str());
    } catch (const Ort::Exception& e) {
        g_session.reset(); g_active_xnnpack.store(false, std::memory_order_release); g_active_nnapi.store(false, std::memory_order_release); g_active_xnnpack_threads = 0;
        throw_runtime(env, std::string("XNNPACK benchmark failed: ") + e.what());
    } catch (const std::exception& e) {
        g_session.reset(); g_active_xnnpack.store(false, std::memory_order_release); g_active_nnapi.store(false, std::memory_order_release); g_active_xnnpack_threads = 0;
        throw_runtime(env, std::string("XNNPACK benchmark failed: ") + e.what());
    }
    return nullptr;
}
'''
p.write_text(s)

p = Path("app/src/main/java/com/CodeBySonu/VoxSherpa/system/TtsDiagnosticsActivity.java")
s = p.read_text()
s = replace_once(s,
    'import com.CodeBySonu.VoxSherpa.vietnamese.VietnameseKokoroEngine;\n',
    'import com.CodeBySonu.VoxSherpa.vietnamese.VietnameseKokoroEngine;\n\nimport org.json.JSONObject;\n',
    "diagnostics JSONObject import")
s = replace_once(s, '    private Button xnnpackBackend;\n', '    private Button xnnpackBackend;\n    private Button xnnpackBenchmark;\n', "diagnostics benchmark field")
s = replace_once(s,
'''            root.addView(xnnpackBackend, new LinearLayout.LayoutParams(
                    LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT));
        }
''',
'''            root.addView(xnnpackBackend, new LinearLayout.LayoutParams(
                    LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT));
            xnnpackBenchmark = new Button(this);
            xnnpackBenchmark.setText("XNNPACK benchmark • CPU / 1 / 2 / 4 / 6 / 8 threads");
            xnnpackBenchmark.setContentDescription(
                    "Automatically benchmark CPU and XNNPACK with one, two, four, six, and eight threads. The fastest safe backend and best XNNPACK thread count will be saved.");
            xnnpackBenchmark.setOnClickListener(v -> runXnnpackBenchmark());
            root.addView(xnnpackBenchmark, new LinearLayout.LayoutParams(
                    LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT));
        }
''', "diagnostics benchmark button")
s = replace_once(s,
'''        xnnpackBackend.setText(xnnRequested
                ? (engine.isXnnpackActive() ? "XNNPACK • ORT 1.26.0 — SELECTED / ACTIVE"
                        : "XNNPACK • ORT 1.26.0 — SELECTED / FALLBACK CPU")
                : "XNNPACK • ORT 1.26.0 — use this mode");
''',
'''        int xnnThreads = engine.selectedXnnpackThreads(this);
        xnnpackBackend.setText(xnnRequested
                ? (engine.isXnnpackActive() ? "XNNPACK • ORT 1.26.0 • " + xnnThreads + " threads — SELECTED / ACTIVE"
                        : "XNNPACK • ORT 1.26.0 • " + xnnThreads + " threads — SELECTED / FALLBACK CPU")
                : "XNNPACK • ORT 1.26.0 • " + xnnThreads + " threads — use this mode");
''', "diagnostics xnnpack label")
ui_method = r'''    private void runXnnpackBenchmark() {
        if (!BuildConfig.KOKORO_XNNPACK_AB || xnnpackBenchmark == null) return;
        VietnameseKokoroEngine engine = VietnameseKokoroEngine.getInstance();
        xnnpackBenchmark.setEnabled(false);
        if (cpuBackend != null) cpuBackend.setEnabled(false);
        if (xnnpackBackend != null) xnnpackBackend.setEnabled(false);
        xnnpackBenchmark.setText("XNNPACK benchmark running…");
        TtsDiagnostics.info(this, "benchmark", "xnnpack_ui_requested",
                "User started automatic CPU vs XNNPACK 1/2/4/6/8 thread benchmark.");
        new Thread(() -> {
            try {
                String json = engine.benchmarkXnnpackThreads(this);
                JSONObject result = new JSONObject(json);
                int bestThreads = result.getInt("bestXnnpackThreads");
                double ratio = result.getDouble("bestXnnpackVsCpuRatio");
                String recommended = result.getString("recommendedBackend");
                runOnUiThread(() -> {
                    xnnpackBenchmark.setEnabled(true);
                    if (cpuBackend != null) cpuBackend.setEnabled(true);
                    if (xnnpackBackend != null) xnnpackBackend.setEnabled(true);
                    xnnpackBenchmark.setText("XNNPACK benchmark • CPU / 1 / 2 / 4 / 6 / 8 threads");
                    refresh();
                    String message = "Benchmark complete. Best XNNPACK=" + bestThreads
                            + " threads, ratio vs CPU=" + String.format(java.util.Locale.US, "%.2f", ratio)
                            + ", selected=" + recommended + ".";
                    Toast.makeText(this, message, Toast.LENGTH_LONG).show();
                });
            } catch (Throwable t) {
                TtsDiagnostics.error(this, "benchmark", "xnnpack_ui_failed", t.toString(), t);
                runOnUiThread(() -> {
                    xnnpackBenchmark.setEnabled(true);
                    if (cpuBackend != null) cpuBackend.setEnabled(true);
                    if (xnnpackBackend != null) xnnpackBackend.setEnabled(true);
                    xnnpackBenchmark.setText("XNNPACK benchmark • CPU / 1 / 2 / 4 / 6 / 8 threads");
                    refresh();
                    Toast.makeText(this, "XNNPACK benchmark failed. CPU was restored; see TTS Logs.", Toast.LENGTH_LONG).show();
                });
            }
        }, "KokoroVi-XNNPACK-Benchmark").start();
    }

'''
s = replace_once(s, '    private void refreshProviderButton() {\n', ui_method + '    private void refreshProviderButton() {\n', "diagnostics benchmark method")
p.write_text(s)

print("Applied XNNPACK automatic thread benchmark patch.")
