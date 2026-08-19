#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def replace_once(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    if old not in text:
        raise SystemExit(f"Expected text not found in {path}: {old[:120]!r}")
    text = text.replace(old, new, 1)
    path.write_text(text, encoding="utf-8")


def replace_between(path: Path, start: str, end: str, replacement: str) -> None:
    text = path.read_text(encoding="utf-8")
    a = text.find(start)
    if a < 0:
        raise SystemExit(f"Start marker not found in {path}: {start!r}")
    b = text.find(end, a)
    if b < 0:
        raise SystemExit(f"End marker not found in {path}: {end!r}")
    text = text[:a] + replacement + text[b:]
    path.write_text(text, encoding="utf-8")


engine = ROOT / "app/src/main/java/com/CodeBySonu/VoxSherpa/vietnamese/VietnameseKokoroEngine.java"
cpp = ROOT / "app/src/main/cpp/kokoro_vi/kokoro_vi_jni.cpp"
prepare = ROOT / "scripts/prepare_kokoro_vi_stage1.sh"
diag = ROOT / "app/src/main/java/com/CodeBySonu/VoxSherpa/system/TtsDiagnosticsActivity.java"

# A 21-token shape seen in real TalkBack diagnostics: representative without monopolizing ORT for several seconds.
replace_once(
    engine,
    '    private static final String WARMUP_TEXT =\n            "Xin chào, giọng đọc tiếng Việt đang được khởi động để phản hồi nhanh và ổn định hơn.";',
    '    private static final String WARMUP_TEXT = "Giọng nói chính.";'
)

replace_between(
    engine,
    '    private synchronized void prewarmBlocking(Context context, String reason) throws Exception {',
    '    public void cancel() {',
    '''    private synchronized void prewarmBlocking(Context context, String reason) throws Exception {
        prepare(context);
        if (modelWarm) return;
        if (!warmCurrentSession(context, "prewarm:" + reason)) {
            TtsDiagnostics.warn(context, "prewarm", "warmup_interrupted",
                    "reason=" + reason + ", session remains usable; the first successful real request will confirm warm state.");
        }
    }

    private boolean warmCurrentSession(Context context, String purpose) throws Exception {
        long started = System.nanoTime();
        float[] audio = synthesizeChunk(WARMUP_TEXT, VietnameseKokoroVoice.DEFAULT, 1.0f, purpose);
        boolean warmed = audio != null && audio.length > 0;
        if (warmed) modelWarm = true;
        TtsDiagnostics.info(context, "prewarm", "representative_inference",
                "purpose=" + purpose + ", voice=" + VietnameseKokoroVoice.DEFAULT.id
                        + ", textChars=" + WARMUP_TEXT.length()
                        + ", audioSamples=" + (audio == null ? 0 : audio.length)
                        + ", warmed=" + warmed + ", elapsedMs=" + elapsedMs(started));
        return warmed;
    }

'''
)

replace_once(
    engine,
    '        if (!modelWarm) warmCurrentSession(app, "provider_switch");',
    '        if (!modelWarm && !warmCurrentSession(app, "provider_switch")) {\n            throw new IllegalStateException("Vietnamese Kokoro provider warm-up was interrupted.");\n        }'
)

replace_once(
    engine,
    '        TtsDiagnostics.info(app, "benchmark", "cpu_start",\n                "Testing ORT default plus explicit CPU thread counts up to 8 across short/medium/long workloads; "\n                        + "1 warmup + 3 measured runs per workload. TTS requests may block during this benchmark.");',
    '        TtsDiagnostics.info(app, "benchmark", "cpu_start",\n                "Testing ORT default/3/4/6 across short/medium TalkBack workloads; "\n                        + "1 warmup + 2 measured runs per workload. TTS requests may briefly block during this benchmark.");'
)
replace_once(
    engine,
    '            json = nativeBridge.benchmarkCpuThreads(assets.model.getAbsolutePath(), ids, style, 1.0f, 1, 3);',
    '            json = nativeBridge.benchmarkCpuThreads(assets.model.getAbsolutePath(), ids, style, 1.0f, 1, 2);'
)
replace_once(
    engine,
    '            if (!modelWarm) warmCurrentSession(app, "benchmark_restore_nnapi");\n        } else {\n            warmCurrentSession(app, "benchmark_selected_cpu");\n        }',
    '            if (!modelWarm && !warmCurrentSession(app, "benchmark_restore_nnapi")) {\n                throw new IllegalStateException("Vietnamese Kokoro benchmark restore warm-up was interrupted.");\n            }\n        } else if (!warmCurrentSession(app, "benchmark_selected_cpu")) {\n            throw new IllegalStateException("Vietnamese Kokoro selected CPU session warm-up was interrupted.");\n        }'
)

# Any successful real inference is definitive evidence that the current session is warm.
replace_once(
    engine,
    '        float[] audio = nativeBridge.synthesize(ids, style, safeSpeed);\n        long nativeCallMs = elapsedMs(nativeStart);',
    '        float[] audio = nativeBridge.synthesize(ids, style, safeSpeed);\n        long nativeCallMs = elapsedMs(nativeStart);\n        if (audio != null && audio.length > 0 && !modelWarm) {\n            modelWarm = true;\n            if (appContext != null) {\n                TtsDiagnostics.info(appContext, "prewarm", "session_confirmed_warm",\n                        "purpose=" + purpose + ", tokenIds=" + ids.length\n                                + ", nativeCallMs=" + nativeCallMs);\n            }\n        }'
)

# Two-sample median is the mean of the middle pair; this lets the light benchmark use only two measured runs.
replace_once(
    cpp,
    '''int64_t median_us(std::vector<int64_t> values) {
    if (values.empty()) return 0;
    std::sort(values.begin(), values.end());
    return values[values.size() / 2];
}''',
    '''int64_t median_us(std::vector<int64_t> values) {
    if (values.empty()) return 0;
    std::sort(values.begin(), values.end());
    const size_t mid = values.size() / 2;
    if ((values.size() & 1u) != 0u) return values[mid];
    return (values[mid - 1] + values[mid]) / 2;
}'''
)

replace_between(
    cpp,
    '        const int warmups = std::max(1, static_cast<int>(warmup_runs));',
    '        const std::string out = json.str();',
    '''        const int warmups = std::max(1, static_cast<int>(warmup_runs));
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
        for (int candidate : candidates) {
            CandidateBenchmark result;
            result.threads = candidate;
            auto load_start = Clock::now();
            auto session = create_session(path, candidate, false);
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

        g_session = create_session(path, best_threads, false);
        g_active_threads = best_threads;
        g_active_nnapi.store(false, std::memory_order_release);

        std::ostringstream json;
        json << "{\\\"bestThreads\\\":" << best_threads
             << ",\\\"rawBestThreads\\\":" << raw_best_threads
             << ",\\\"preferredOrtDefaultForAffinity\\\":"
             << (preferred_ort_default_for_affinity ? "true" : "false")
             << ",\\\"onlineCpus\\\":" << online_cpus
             << ",\\\"workloadTokenCounts\\\":["
             << workloads[0].size() << "," << workloads[1].size() << "]"
             << ",\\\"candidates\\\":[";
        for (size_t i = 0; i < results.size(); ++i) {
            if (i != 0) json << ",";
            const CandidateBenchmark& r = results[i];
            json << "{\\\"threads\\\":" << r.threads
                 << ",\\\"autoAffinity\\\":" << (r.threads == 0 ? "true" : "false")
                 << ",\\\"loadMs\\\":" << r.load_ms
                 << ",\\\"shortMedianUs\\\":" << r.median_us[0]
                 << ",\\\"mediumMedianUs\\\":" << r.median_us[1]
                 << ",\\\"scoreUs\\\":" << r.score_us << "}";
        }
        json << "]}";
'''
)

# Keep 1.17.1 as the proven runtime; CI still builds 1.28.0 side-by-side for comparison.
replace_once(prepare, 'ORT_VERSION="${ORT_VERSION:-1.28.0}"', 'ORT_VERSION="${ORT_VERSION:-1.17.1}"')

if diag.exists():
    text = diag.read_text(encoding="utf-8")
    old = "User started default/3/4/5/6 CPU benchmark with 2 warmups + 5 measured runs per mode."
    new = "User started lightweight ORT default/3/4/6 CPU benchmark across short/medium TalkBack workloads."
    if old in text:
        diag.write_text(text.replace(old, new, 1), encoding="utf-8")

print("Applied lightweight warm-up, CPU benchmark, auto-affinity, and ORT 1.17.1 default follow-up.")
