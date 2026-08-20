#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CPP = ROOT / "app/src/main/cpp/kokoro_vi/kokoro_vi_jni.cpp"
ASSETS = ROOT / "app/src/main/java/com/CodeBySonu/VoxSherpa/vietnamese/VietnameseKokoroAssetStore.java"
ENGINE = ROOT / "app/src/main/java/com/CodeBySonu/VoxSherpa/vietnamese/VietnameseKokoroEngine.java"
ACTIVITY = ROOT / "app/src/main/java/com/CodeBySonu/VoxSherpa/system/TtsDiagnosticsActivity.java"


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{label}: expected exactly one match, found {count}")
    return text.replace(old, new, 1)


# Native: keep the existing JNI signature. cpu_threads=-2 is an experiment-only
# sentinel meaning ORT default/auto-affinity plus session.dynamic_block_base=2.
cpp = CPP.read_text(encoding="utf-8")
cpp = replace_once(
    cpp,
    "std::unique_ptr<Ort::Session> create_session(const std::string& path, int cpu_threads, bool use_nnapi) {\n",
    "std::unique_ptr<Ort::Session> create_session(const std::string& path, int cpu_threads, bool use_nnapi, int dynamic_block_base = 0) {\n",
    "create_session signature",
)
cpp = replace_once(
    cpp,
    "    options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);\n    if (use_nnapi) append_nnapi(options);\n",
    "    options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);\n"
    "    if (dynamic_block_base > 0) {\n"
    "        const std::string dynamic_value = std::to_string(dynamic_block_base);\n"
    "        options.AddConfigEntry(\"session.dynamic_block_base\", dynamic_value.c_str());\n"
    "    }\n"
    "    if (use_nnapi) append_nnapi(options);\n",
    "dynamic block config",
)
cpp = replace_once(
    cpp,
    "    const bool requested_nnapi = use_nnapi == JNI_TRUE;\n    terminate_active_run();\n",
    "    const bool requested_nnapi = use_nnapi == JNI_TRUE;\n"
    "    const bool tuned_dynamic2 = cpu_threads == -2;\n"
    "    const int requested_threads = tuned_dynamic2 ? 0 : static_cast<int>(cpu_threads);\n"
    "    const int dynamic_block_base = tuned_dynamic2 ? 2 : 0;\n"
    "    terminate_active_run();\n",
    "createEngine sentinel",
)
cpp = cpp.replace("create_session(path, cpu_threads, true)", "create_session(path, requested_threads, true, dynamic_block_base)")
cpp = cpp.replace("create_session(path, cpu_threads, false)", "create_session(path, requested_threads, false, dynamic_block_base)")
cpp = replace_once(
    cpp,
    "        g_active_threads = cpu_threads;\n        return JNI_TRUE;\n",
    "        g_active_threads = requested_threads;\n"
    "        LOGI(\"Selective INT8 experiment session: model=%s cpuThreads=%d dynamicBlockBase=%d\",\n"
    "             path.c_str(), requested_threads, dynamic_block_base);\n"
    "        return JNI_TRUE;\n",
    "active thread assignment",
)
CPP.write_text(cpp, encoding="utf-8")


# Asset store: bundle the quantized model, but copy it lazily only when INT8 is selected.
assets = ASSETS.read_text(encoding="utf-8")
assets = replace_once(
    assets,
    '    private static final String MODEL = ROOT + "/kokoro_vi.onnx";\n',
    '    private static final String MODEL = ROOT + "/kokoro_vi.onnx";\n'
    '    private static final String INT8_MODEL = ROOT + "/kokoro_vi_int8.onnx";\n',
    "INT8 asset constant",
)
assets = replace_once(
    assets,
    "            boolean assets = assetExists(context, MODEL)\n                    && assetExists(context, DICTIONARY)\n                    && assetExists(context, CONFIG);\n",
    "            boolean assets = assetExists(context, MODEL)\n"
    "                    && assetExists(context, INT8_MODEL)\n"
    "                    && assetExists(context, DICTIONARY)\n"
    "                    && assetExists(context, CONFIG);\n",
    "bundle probe",
)
insert_anchor = "    public static synchronized File ensureVoice(Context context, VietnameseKokoroVoice voice) throws Exception {\n"
int8_method = '''    public static synchronized File ensureInt8Model(Context context) throws Exception {
        if (context == null) throw new IllegalArgumentException("Context required.");
        if (!assetExists(context, INT8_MODEL)) {
            throw new IllegalStateException("Missing bundled selective INT8 Kokoro model.");
        }
        File root = ensureRoot(context);
        return copyIfNeeded(context, INT8_MODEL, new File(root, "kokoro_vi_int8.onnx"));
    }

'''
assets = replace_once(assets, insert_anchor, int8_method + insert_anchor, "ensureInt8Model insertion")
ASSETS.write_text(assets, encoding="utf-8")


# Engine: add model variants, CPU-only switching, an automatic speed benchmark,
# and keep INT8 opt-in until the user has listened to it.
engine = ENGINE.read_text(encoding="utf-8")
engine = replace_once(engine, "import org.json.JSONObject;\n", "import org.json.JSONArray;\nimport org.json.JSONObject;\n", "JSONArray import")
engine = replace_once(
    engine,
    '    private static final String PREF_NNAPI_ENABLED = "nnapi_enabled";\n',
    '    private static final String PREF_NNAPI_ENABLED = "nnapi_enabled";\n'
    '    private static final String PREF_MODEL_VARIANT = "selective_int8_model_variant";\n'
    '    public static final String MODE_FP32_BASELINE = "fp32_baseline";\n'
    '    public static final String MODE_FP32_DYNAMIC2 = "fp32_dynamic2";\n'
    '    public static final String MODE_INT8_DYNAMIC2 = "int8_dynamic2";\n',
    "variant constants",
)
engine = replace_once(
    engine,
    "    private volatile int activeCpuThreads = 0;\n",
    "    private volatile int activeCpuThreads = 0;\n"
    "    private volatile String activeModelVariant = MODE_FP32_BASELINE;\n"
    "    private volatile int activeDynamicBlockBase = 0;\n",
    "active variant fields",
)
engine = replace_once(
    engine,
    "    public boolean isNnapiActive() {\n        return activeNnapi;\n    }\n\n",
    '''    public boolean isNnapiActive() {
        return activeNnapi;
    }

    public String requestedModelVariant(Context context) {
        if (context == null) return MODE_FP32_BASELINE;
        String mode = context.getSharedPreferences(PERF_PREFS, Context.MODE_PRIVATE)
                .getString(PREF_MODEL_VARIANT, MODE_FP32_BASELINE);
        if (MODE_FP32_DYNAMIC2.equals(mode) || MODE_INT8_DYNAMIC2.equals(mode)) return mode;
        return MODE_FP32_BASELINE;
    }

    public boolean isInt8Active() {
        return MODE_INT8_DYNAMIC2.equals(activeModelVariant);
    }

''',
    "variant getters",
)
engine = replace_once(
    engine,
    '                    + ", savedCpuThreads=" + saved + " (0=ORT default/auto-affinity)"\n                    + ", cachedVoiceStyles=" + voiceStyleCache.size();\n',
    '                    + ", savedCpuThreads=" + saved + " (0=ORT default/auto-affinity)"\n'
    '                    + ", modelVariant=" + activeModelVariant\n'
    '                    + ", dynamicBlockBase=" + activeDynamicBlockBase\n'
    '                    + ", cachedVoiceStyles=" + voiceStyleCache.size();\n',
    "performance state",
)

variant_methods_anchor = "    public boolean synthesizeStreaming(Context context, String text, VietnameseKokoroVoice voice,\n"
variant_methods = r'''    public synchronized void setModelVariant(Context context, String mode) throws Exception {
        Context app = context.getApplicationContext();
        if (!MODE_FP32_BASELINE.equals(mode) && !MODE_FP32_DYNAMIC2.equals(mode)
                && !MODE_INT8_DYNAMIC2.equals(mode)) {
            throw new IllegalArgumentException("Unknown Kokoro model variant: " + mode);
        }
        appContext = app;
        SharedPreferences prefs = app.getSharedPreferences(PERF_PREFS, Context.MODE_PRIVATE);
        prefs.edit()
                .putString(PREF_MODEL_VARIANT, mode)
                .putBoolean(PREF_NNAPI_ENABLED, false)
                .putInt(PREF_CPU_THREADS, 0)
                .apply();
        cancel();
        if (modelReady) nativeBridge.destroyEngine();
        modelReady = false;
        modelWarm = false;
        activeNnapi = false;
        activeCpuThreads = 0;
        prepare(app);
        if (!modelWarm && !warmCurrentSession(app, "model_variant_switch:" + mode)) {
            throw new IllegalStateException("Vietnamese Kokoro model variant warm-up was interrupted.");
        }
        TtsDiagnostics.info(app, "quantization", "variant_selected",
                "mode=" + mode + ", active=" + activeModelVariant + ", " + performanceState(app));
    }

    private long[] benchmarkResizeIds(long[] base, int targetSize) {
        int target = Math.max(3, Math.min(512, targetSize));
        if (base == null || base.length < 3) return base;
        long[] out = new long[target];
        out[0] = base[0];
        out[target - 1] = base[base.length - 1];
        int cursor = 1;
        for (int i = 1; i < target - 1; i++) {
            out[i] = base[cursor++];
            if (cursor >= base.length - 1) cursor = 1;
        }
        return out;
    }

    private JSONObject benchmarkCurrentVariant(String mode, long[] baseIds, float[] style) throws Exception {
        final int[] tokenCounts = {21, 50, 71, 110};
        final int[] weights = {4, 3, 2, 1};
        JSONArray medians = new JSONArray();
        long weighted = 0L;
        for (int w = 0; w < tokenCounts.length; w++) {
            long[] ids = benchmarkResizeIds(baseIds, tokenCounts[w]);
            nativeBridge.synthesize(ids, style, 1.0f); // workload-specific warmup
            long[] samples = new long[2];
            for (int r = 0; r < 2; r++) {
                long started = System.nanoTime();
                float[] audio = nativeBridge.synthesize(ids, style, 1.0f);
                long elapsedUs = (System.nanoTime() - started) / 1000L;
                if (audio == null || audio.length == 0) {
                    throw new IllegalStateException("Empty benchmark waveform for " + mode + " tokens=" + tokenCounts[w]);
                }
                samples[r] = elapsedUs;
            }
            long median = (samples[0] + samples[1]) / 2L;
            medians.put(median);
            weighted += median * weights[w];
        }
        JSONObject out = new JSONObject();
        out.put("mode", mode);
        out.put("mediansUs", medians);
        out.put("weightedScoreUs", weighted);
        return out;
    }

    public synchronized String benchmarkModelVariants(Context context) throws Exception {
        Context app = context.getApplicationContext();
        appContext = app;
        long totalStart = System.nanoTime();
        TtsDiagnostics.info(app, "quantization", "benchmark_start",
                "A/B/C CPU benchmark: FP32 baseline, FP32 dynamic_block_base=2, selective INT8 dynamic_block_base=2. "
                        + "Workloads=21/50/71/110 tokens, weights=4/3/2/1, one workload warmup + two measured runs. "
                        + "INT8 will NOT be auto-selected; voice quality must be listened manually first.");

        // Ensure G2P/vocab/voice assets exist before constructing representative input.
        prepare(app);
        String phonemes = phonemize(WARMUP_TEXT);
        long[] baseIds = tokenIds(phonemes);
        float[] style = selectStyle(VietnameseKokoroVoice.DEFAULT, Math.max(1, phonemes.length()));

        setModelVariant(app, MODE_FP32_BASELINE);
        JSONObject baselineBefore = benchmarkCurrentVariant(MODE_FP32_BASELINE, baseIds, style);
        setModelVariant(app, MODE_FP32_DYNAMIC2);
        JSONObject fp32Dynamic2 = benchmarkCurrentVariant(MODE_FP32_DYNAMIC2, baseIds, style);
        setModelVariant(app, MODE_INT8_DYNAMIC2);
        JSONObject int8Dynamic2 = benchmarkCurrentVariant(MODE_INT8_DYNAMIC2, baseIds, style);
        setModelVariant(app, MODE_FP32_BASELINE);
        JSONObject baselineAfter = benchmarkCurrentVariant(MODE_FP32_BASELINE, baseIds, style);

        long baselineScore = (baselineBefore.getLong("weightedScoreUs")
                + baselineAfter.getLong("weightedScoreUs")) / 2L;
        long dynamicScore = fp32Dynamic2.getLong("weightedScoreUs");
        long int8Score = int8Dynamic2.getLong("weightedScoreUs");
        String fastest = MODE_FP32_BASELINE;
        long fastestScore = baselineScore;
        if (dynamicScore < fastestScore) { fastest = MODE_FP32_DYNAMIC2; fastestScore = dynamicScore; }
        if (int8Score < fastestScore) { fastest = MODE_INT8_DYNAMIC2; fastestScore = int8Score; }

        // Leave the lossless tuned FP32 mode active. INT8 remains an explicit listening choice.
        setModelVariant(app, MODE_FP32_DYNAMIC2);

        JSONObject result = new JSONObject();
        result.put("workloadTokenCounts", new JSONArray(new int[]{21, 50, 71, 110}));
        result.put("weights", new JSONArray(new int[]{4, 3, 2, 1}));
        result.put("baselineBefore", baselineBefore);
        result.put("baselineAfter", baselineAfter);
        result.put("baselineReferenceWeightedScoreUs", baselineScore);
        result.put("fp32Dynamic2", fp32Dynamic2);
        result.put("int8Dynamic2", int8Dynamic2);
        result.put("fp32Dynamic2VsBaselineRatio", baselineScore > 0 ? (double) dynamicScore / baselineScore : 1.0);
        result.put("int8Dynamic2VsBaselineRatio", baselineScore > 0 ? (double) int8Score / baselineScore : 1.0);
        result.put("int8VsFp32Dynamic2Ratio", dynamicScore > 0 ? (double) int8Score / dynamicScore : 1.0);
        result.put("fastestMode", fastest);
        result.put("activeAfterBenchmark", MODE_FP32_DYNAMIC2);
        result.put("int8RequiresListeningApproval", true);
        String json = result.toString();
        TtsDiagnostics.info(app, "quantization", "benchmark_complete",
                "elapsedMs=" + elapsedMs(totalStart) + ", result=" + json);
        return json;
    }

'''
engine = replace_once(engine, variant_methods_anchor, variant_methods + variant_methods_anchor, "variant methods insertion")

old_prepare = '''            SharedPreferences prefs = context.getSharedPreferences(PERF_PREFS, Context.MODE_PRIVATE);
            activeCpuThreads = prefs.getInt(PREF_CPU_THREADS, 0);
            boolean requestedNnapi = prefs.getBoolean(PREF_NNAPI_ENABLED, false);
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
                            + ", cpuThreads=" + threadLabel(activeCpuThreads));
'''
new_prepare = '''            SharedPreferences prefs = context.getSharedPreferences(PERF_PREFS, Context.MODE_PRIVATE);
            String requestedVariant = requestedModelVariant(context);
            boolean tunedDynamic2 = MODE_FP32_DYNAMIC2.equals(requestedVariant)
                    || MODE_INT8_DYNAMIC2.equals(requestedVariant);
            File selectedModel = MODE_INT8_DYNAMIC2.equals(requestedVariant)
                    ? VietnameseKokoroAssetStore.ensureInt8Model(context)
                    : assets.model;
            // CPU-only controlled experiment: clear old NNAPI/thread preferences so every
            // variant differs only by quantization and the explicitly named dynamic2 option.
            prefs.edit().putBoolean(PREF_NNAPI_ENABLED, false).putInt(PREF_CPU_THREADS, 0).apply();
            int nativeThreadMode = tunedDynamic2 ? -2 : 0;
            modelReady = nativeBridge.createEngine(selectedModel.getAbsolutePath(), nativeThreadMode, false);
            if (!modelReady) throw new IllegalStateException("Vietnamese Kokoro model failed to load.");
            activeCpuThreads = 0;
            activeNnapi = false;
            activeModelVariant = requestedVariant;
            activeDynamicBlockBase = tunedDynamic2 ? 2 : 0;
            TtsDiagnostics.info(context, "engine", "model_ready",
                    "elapsedMs=" + elapsedMs(started) + ", modelBytes=" + selectedModel.length()
                            + ", provider=CPU, modelVariant=" + activeModelVariant
                            + ", dynamicBlockBase=" + activeDynamicBlockBase
                            + ", cpuThreads=default");
'''
engine = replace_once(engine, old_prepare, new_prepare, "prepare model selection")
ENGINE.write_text(engine, encoding="utf-8")


# Accessible diagnostics UI: hide the old NNAPI control in this CPU-only experiment,
# expose manual listening choices, and repurpose the benchmark button for A/B/C speed.
activity = ACTIVITY.read_text(encoding="utf-8")
activity = replace_once(
    activity,
    "    private Button nnapi;\n",
    "    private Button nnapi;\n"
    "    private Button fp32Baseline;\n"
    "    private Button fp32Dynamic2;\n"
    "    private Button int8Dynamic2;\n",
    "variant button fields",
)
activity = replace_once(
    activity,
    "        root.addView(nnapi, new LinearLayout.LayoutParams(\n                LinearLayout.LayoutParams.MATCH_PARENT,\n                LinearLayout.LayoutParams.WRAP_CONTENT\n        ));\n\n",
    "        root.addView(nnapi, new LinearLayout.LayoutParams(\n"
    "                LinearLayout.LayoutParams.MATCH_PARENT,\n"
    "                LinearLayout.LayoutParams.WRAP_CONTENT\n"
    "        ));\n"
    "        nnapi.setVisibility(android.view.View.GONE);\n\n",
    "hide NNAPI",
)
activity = replace_once(
    activity,
    '        benchmark.setText("CPU benchmark: default / 3 / 4 / 5 / 6");\n',
    '        benchmark.setText("Benchmark FP32 vs selective INT8");\n',
    "benchmark label",
)
activity = replace_once(
    activity,
    '                "Run a deep CPU benchmark for ONNX Runtime default, three, four, five, and six threads. "\n                        + "Each mode uses two warmup runs and five measured runs. Speech may be temporarily blocked while the model is tested.");\n',
    '                "Benchmark FP32 baseline, lossless FP32 dynamic block two, and selective INT8 dynamic block two. "\n'
    '                        + "The benchmark never auto-selects INT8; listen to INT8 manually before deciding whether its voice quality is acceptable.");\n',
    "benchmark description",
)
insert_after_benchmark = '''        root.addView(benchmark, new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.WRAP_CONTENT
        ));

'''
variant_buttons = '''        fp32Baseline = new Button(this);
        fp32Baseline.setContentDescription("Use original FP32 Kokoro with ONNX Runtime default CPU scheduling. This is the production reference quality.");
        fp32Baseline.setOnClickListener(v -> switchModelVariant(VietnameseKokoroEngine.MODE_FP32_BASELINE));
        root.addView(fp32Baseline, new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT));

        fp32Dynamic2 = new Button(this);
        fp32Dynamic2.setContentDescription("Use original lossless FP32 Kokoro with dynamic block base two. Audio precision is unchanged.");
        fp32Dynamic2.setOnClickListener(v -> switchModelVariant(VietnameseKokoroEngine.MODE_FP32_DYNAMIC2));
        root.addView(fp32Dynamic2, new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT));

        int8Dynamic2 = new Button(this);
        int8Dynamic2.setContentDescription("Use selective INT8 Kokoro with MatMul and Gemm weights quantized and dynamic block base two. Compare voice quality carefully against FP32.");
        int8Dynamic2.setOnClickListener(v -> switchModelVariant(VietnameseKokoroEngine.MODE_INT8_DYNAMIC2));
        root.addView(int8Dynamic2, new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT));

'''
activity = replace_once(activity, insert_after_benchmark, insert_after_benchmark + variant_buttons, "variant buttons")
activity = replace_once(
    activity,
    "        refreshProviderButton();\n",
    "        refreshProviderButton();\n        refreshVariantButtons();\n",
    "refresh variants call",
)
refresh_anchor = "    private void toggleNnapi() {\n"
refresh_methods = '''    private void refreshVariantButtons() {
        String mode = VietnameseKokoroEngine.getInstance().requestedModelVariant(this);
        fp32Baseline.setText("FP32 baseline" + (VietnameseKokoroEngine.MODE_FP32_BASELINE.equals(mode) ? " • selected" : ""));
        fp32Dynamic2.setText("FP32 dynamic2" + (VietnameseKokoroEngine.MODE_FP32_DYNAMIC2.equals(mode) ? " • selected" : ""));
        int8Dynamic2.setText("Selective INT8 dynamic2" + (VietnameseKokoroEngine.MODE_INT8_DYNAMIC2.equals(mode) ? " • selected" : ""));
    }

    private void switchModelVariant(String mode) {
        fp32Baseline.setEnabled(false);
        fp32Dynamic2.setEnabled(false);
        int8Dynamic2.setEnabled(false);
        benchmark.setEnabled(false);
        TtsDiagnostics.info(this, "quantization", "variant_ui_requested", "mode=" + mode);
        new Thread(() -> {
            try {
                VietnameseKokoroEngine.getInstance().setModelVariant(this, mode);
                runOnUiThread(() -> {
                    fp32Baseline.setEnabled(true);
                    fp32Dynamic2.setEnabled(true);
                    int8Dynamic2.setEnabled(true);
                    benchmark.setEnabled(true);
                    refresh();
                    Toast.makeText(this, "Kokoro mode active: " + mode + ". Test the same sentence and share the log.", Toast.LENGTH_LONG).show();
                });
            } catch (Throwable t) {
                TtsDiagnostics.error(this, "quantization", "variant_ui_failed", "mode=" + mode + ", error=" + t, t);
                runOnUiThread(() -> {
                    fp32Baseline.setEnabled(true);
                    fp32Dynamic2.setEnabled(true);
                    int8Dynamic2.setEnabled(true);
                    benchmark.setEnabled(true);
                    refresh();
                    Toast.makeText(this, "Model switch failed; see TTS Logs.", Toast.LENGTH_LONG).show();
                });
            }
        }, "KokoroVi-Model-Variant").start();
    }

'''
activity = replace_once(activity, refresh_anchor, refresh_methods + refresh_anchor, "variant UI methods")
activity = replace_once(activity, '        benchmark.setText("CPU benchmark running…");\n', '        benchmark.setText("FP32 / INT8 benchmark running…");\n', "running label")
activity = replace_once(
    activity,
    '        TtsDiagnostics.info(this, "benchmark", "ui_requested",\n                "User started lightweight ORT default/3/4/6 CPU benchmark across short/medium TalkBack workloads.");\n',
    '        TtsDiagnostics.info(this, "quantization", "benchmark_ui_requested",\n                "User started FP32 baseline / FP32 dynamic2 / selective INT8 dynamic2 benchmark.");\n',
    "benchmark ui log",
)
activity = replace_once(
    activity,
    "                VietnameseKokoroEngine.getInstance().benchmarkCpuThreads(this);\n",
    "                VietnameseKokoroEngine.getInstance().benchmarkModelVariants(this);\n",
    "benchmark method call",
)
# The original benchmark label appears twice when the button is restored.
activity = activity.replace('benchmark.setText("CPU benchmark: default / 3 / 4 / 5 / 6");', 'benchmark.setText("Benchmark FP32 vs selective INT8");')
activity = activity.replace('"CPU benchmark complete; fastest CPU mode saved."', '"FP32/INT8 speed benchmark complete. FP32 dynamic2 remains active; listen to INT8 before selecting it."')
activity = activity.replace('"CPU benchmark failed. See TTS Logs."', '"FP32/INT8 benchmark failed. See TTS Logs."')
ACTIVITY.write_text(activity, encoding="utf-8")

print("Selective INT8 A/B/C experiment patch applied successfully")
