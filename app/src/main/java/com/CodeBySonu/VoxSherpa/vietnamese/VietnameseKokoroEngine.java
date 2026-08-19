package com.CodeBySonu.VoxSherpa.vietnamese;

import android.content.Context;
import android.content.SharedPreferences;

import com.CodeBySonu.VoxSherpa.system.TtsDiagnostics;

import org.json.JSONObject;

import java.io.ByteArrayOutputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/** Proven Kokoro-Vietnamese pipeline ported from the Pocodo engine. */
public final class VietnameseKokoroEngine {
    public static final int SAMPLE_RATE = 24000;
    private static final int MAX_PHONEMES = 510;
    private static final int STYLE_SIZE = 256;
    private static final String PERF_PREFS = "kokoro_vi_perf";
    private static final String PREF_CPU_THREADS = "cpu_threads";
    private static final Pattern SENTENCE_BOUNDARY = Pattern.compile("[.!?…]+(?:[\\\"”’)]*)");
    private static final Pattern WORD_TOKEN = Pattern.compile("^[A-Za-zÀ-ỹĐđ]+(?:[-'][A-Za-zÀ-ỹĐđ]+)*$");
    private static final Pattern TOKEN_PATTERN = Pattern.compile("[A-Za-zÀ-ỹĐđ]+(?:[-'][A-Za-zÀ-ỹĐđ]+)*|\\s+|.");
    private static final Pattern WHITESPACE = Pattern.compile("\\s+");
    private static final Pattern GI_ACCENT = Pattern.compile("^g[iìíỉĩị]");
    private static volatile VietnameseKokoroEngine instance;

    private final VietnameseKokoroNative nativeBridge = new VietnameseKokoroNative();
    private final AtomicLong generation = new AtomicLong(0L);
    private final AtomicBoolean prewarmQueued = new AtomicBoolean(false);
    private final ExecutorService prewarmExecutor = Executors.newSingleThreadExecutor(r -> {
        Thread t = new Thread(r, "KokoroVi-Prewarm");
        t.setPriority(Thread.NORM_PRIORITY + 1);
        return t;
    });

    private VietnameseKokoroAssetStore.Paths assets;
    private long g2pHandle;
    private volatile boolean modelReady;
    private volatile boolean modelWarm;
    private volatile int activeCpuThreads = 6;
    private Map<Character, Long> vocab = new HashMap<>();
    private float[] diemTrinhStyles;
    private Context appContext;

    public interface PcmConsumer { boolean onPcm(byte[] pcm); }

    private VietnameseKokoroEngine() {}

    public static VietnameseKokoroEngine getInstance() {
        if (instance == null) {
            synchronized (VietnameseKokoroEngine.class) {
                if (instance == null) instance = new VietnameseKokoroEngine();
            }
        }
        return instance;
    }

    public static boolean isBundled(Context context) {
        return VietnameseKokoroAssetStore.isBundled(context);
    }

    public boolean isReady() {
        return modelReady && modelWarm && g2pHandle != 0L;
    }

    public String performanceState(Context context) {
        int saved = context.getSharedPreferences(PERF_PREFS, Context.MODE_PRIVATE).getInt(PREF_CPU_THREADS, 6);
        return "ready=" + isReady() + ", modelReady=" + modelReady + ", warm=" + modelWarm
                + ", g2pReady=" + (g2pHandle != 0L) + ", activeCpuThreads=" + activeCpuThreads
                + ", savedCpuThreads=" + saved + " (0=ORT default)";
    }

    public void prewarmAsync(Context context, String reason) {
        if (context == null || !isBundled(context)) return;
        final Context app = context.getApplicationContext();
        if (isReady()) {
            TtsDiagnostics.info(app, "prewarm", "already_hot",
                    "reason=" + reason + ", " + performanceState(app));
            return;
        }
        if (!prewarmQueued.compareAndSet(false, true)) {
            TtsDiagnostics.info(app, "prewarm", "already_queued", "reason=" + reason);
            return;
        }
        TtsDiagnostics.info(app, "prewarm", "queued", "reason=" + reason);
        prewarmExecutor.execute(() -> {
            long started = System.nanoTime();
            try {
                prewarmBlocking(app, reason);
                TtsDiagnostics.info(app, "prewarm", "complete",
                        "reason=" + reason + ", elapsedMs=" + elapsedMs(started)
                                + ", " + performanceState(app));
            } catch (Throwable t) {
                TtsDiagnostics.error(app, "prewarm", "failed",
                        "reason=" + reason + ", elapsedMs=" + elapsedMs(started), t);
            } finally {
                prewarmQueued.set(false);
            }
        });
    }

    private synchronized void prewarmBlocking(Context context, String reason) throws Exception {
        prepare(context);
        if (modelWarm) return;
        long started = System.nanoTime();
        float[] audio = synthesizeChunk("xin", VietnameseKokoroVoice.DIEM_TRINH, 1.0f, "prewarm");
        modelWarm = audio != null && audio.length > 0;
        TtsDiagnostics.info(context, "prewarm", "tiny_inference",
                "reason=" + reason + ", audioSamples=" + (audio == null ? 0 : audio.length)
                        + ", elapsedMs=" + elapsedMs(started));
    }

    public void cancel() {
        long newGeneration = generation.incrementAndGet();
        if (appContext != null) {
            TtsDiagnostics.info(appContext, "engine", "cancel",
                    "generation=" + newGeneration + ", modelReady=" + modelReady
                            + ", modelWarm=" + modelWarm + ", g2pReady=" + (g2pHandle != 0L));
        }
    }

    public boolean synthesizeStreaming(Context context, String text, VietnameseKokoroVoice voice,
                                       float speed, PcmConsumer consumer) {
        if (context == null || text == null || text.trim().isEmpty() || consumer == null) return false;
        final long requestGeneration = generation.incrementAndGet();
        final long startedAt = System.nanoTime();
        try {
            long prepareStart = System.nanoTime();
            prepare(context.getApplicationContext());
            long prepareMs = elapsedMs(prepareStart);
            boolean emitted = false;
            boolean firstPcm = true;
            int chunkCount = 0;
            for (String chunk : splitWithBoundary(text)) {
                if (requestGeneration != generation.get()) return emitted;
                long splitStart = System.nanoTime();
                List<String> safeChunks = splitLongChunk(chunk);
                long splitMs = elapsedMs(splitStart);
                for (String safeChunk : safeChunks) {
                    if (requestGeneration != generation.get()) return emitted;
                    chunkCount++;
                    float[] audio = synthesizeChunk(safeChunk, voice, speed, "request");
                    if (requestGeneration != generation.get()) return emitted;
                    if (audio != null && audio.length > 0) {
                        long pcmStart = System.nanoTime();
                        byte[] pcm = toPcm16(audio);
                        long pcmMs = elapsedMs(pcmStart);
                        if (firstPcm) {
                            firstPcm = false;
                            TtsDiagnostics.info(context, "latency", "first_pcm_ready",
                                    "generation=" + requestGeneration + ", requestToPcmMs=" + elapsedMs(startedAt)
                                            + ", prepareMs=" + prepareMs + ", splitCheckMs=" + splitMs
                                            + ", pcmConvertMs=" + pcmMs + ", bytes=" + pcm.length);
                        }
                        emitted = true;
                        if (!consumer.onPcm(pcm)) return emitted;
                    }
                }
            }
            TtsDiagnostics.info(context, "engine", "stream_done",
                    "generation=" + requestGeneration + ", chunks=" + chunkCount
                            + ", emitted=" + emitted + ", elapsedMs=" + elapsedMs(startedAt));
            return emitted;
        } catch (Throwable t) {
            TtsDiagnostics.error(context, "engine", "synthesis_exception",
                    "generation=" + requestGeneration + ", voice=" + voice.id
                            + ", speed=" + speed + ", chars=" + text.length(), t);
            return false;
        }
    }

    /** Keep the large ONNX session alive across TTS service reconnects. */
    public void retainAcrossServiceDestroy(Context context) {
        if (context != null) {
            TtsDiagnostics.info(context, "engine", "session_retained",
                    "TTS service destroyed; keeping Vietnamese ONNX/G2P session hot in process. "
                            + performanceState(context));
        }
    }

    /** Release only when Android explicitly reports serious memory pressure. */
    public synchronized void releaseForMemoryPressure(Context context, String reason) {
        if (context != null) appContext = context.getApplicationContext();
        TtsDiagnostics.warn(appContext, "engine", "memory_pressure_release",
                "reason=" + reason + ", " + (appContext == null ? "" : performanceState(appContext)));
        releaseInternal();
    }

    public synchronized void release() {
        releaseInternal();
    }

    private void releaseInternal() {
        generation.incrementAndGet();
        try {
            if (g2pHandle != 0L) nativeBridge.destroyG2p(g2pHandle);
        } catch (Throwable t) {
            if (appContext != null) TtsDiagnostics.error(appContext, "engine", "destroy_g2p_failed", t.toString(), t);
        }
        g2pHandle = 0L;
        try {
            if (modelReady) nativeBridge.destroyEngine();
        } catch (Throwable t) {
            if (appContext != null) TtsDiagnostics.error(appContext, "engine", "destroy_model_failed", t.toString(), t);
        }
        modelReady = false;
        modelWarm = false;
        assets = null;
        vocab.clear();
        diemTrinhStyles = null;
        if (appContext != null) TtsDiagnostics.info(appContext, "engine", "released", "Vietnamese engine resources released.");
    }

    public synchronized String benchmarkCpuThreads(Context context) throws Exception {
        Context app = context.getApplicationContext();
        appContext = app;
        long totalStart = System.nanoTime();
        TtsDiagnostics.info(app, "benchmark", "cpu_start",
                "Testing ORT default/4/6 threads; TTS requests may block during this benchmark.");
        prepare(app);
        String phonemes = phonemize("xin chào");
        long[] ids = tokenIds(phonemes);
        float[] style = selectStyle(VietnameseKokoroVoice.DIEM_TRINH, phonemes.length());
        String json = nativeBridge.benchmarkCpuThreads(
                assets.model.getAbsolutePath(), ids, style, 1.0f, 1, 3);
        JSONObject result = new JSONObject(json);
        int best = result.getInt("bestThreads");
        app.getSharedPreferences(PERF_PREFS, Context.MODE_PRIVATE)
                .edit().putInt(PREF_CPU_THREADS, best).apply();
        activeCpuThreads = best;
        modelReady = true;
        modelWarm = true;
        TtsDiagnostics.info(app, "benchmark", "cpu_complete",
                "elapsedMs=" + elapsedMs(totalStart) + ", result=" + json
                        + ", selected=" + threadLabel(best));
        return json;
    }

    private synchronized void prepare(Context context) throws Exception {
        appContext = context.getApplicationContext();
        if (!VietnameseKokoroNative.isAvailable()) {
            IllegalStateException error = new IllegalStateException(VietnameseKokoroNative.loadError());
            TtsDiagnostics.error(context, "engine", "native_unavailable", error.getMessage(), error);
            throw error;
        }
        if (assets == null) {
            long started = System.nanoTime();
            assets = VietnameseKokoroAssetStore.ensure(context);
            vocab = loadVocab(context);
            TtsDiagnostics.info(context, "engine", "assets_and_vocab_ready",
                    "vocabEntries=" + vocab.size() + ", elapsedMs=" + elapsedMs(started));
        }
        if (g2pHandle == 0L) {
            long started = System.nanoTime();
            g2pHandle = nativeBridge.createG2p(assets.dictionary.getAbsolutePath());
            if (g2pHandle == 0L) throw new IllegalStateException("Vietnamese G2P failed to initialize.");
            TtsDiagnostics.info(context, "engine", "g2p_ready", "elapsedMs=" + elapsedMs(started));
        }
        if (!modelReady) {
            long started = System.nanoTime();
            SharedPreferences prefs = context.getSharedPreferences(PERF_PREFS, Context.MODE_PRIVATE);
            activeCpuThreads = prefs.getInt(PREF_CPU_THREADS, 6);
            modelReady = nativeBridge.createEngine(assets.model.getAbsolutePath(), activeCpuThreads);
            if (!modelReady) throw new IllegalStateException("Vietnamese Kokoro model failed to load.");
            TtsDiagnostics.info(context, "engine", "model_ready",
                    "elapsedMs=" + elapsedMs(started) + ", modelBytes=" + assets.model.length()
                            + ", cpuThreads=" + threadLabel(activeCpuThreads));
        }
    }

    private Map<Character, Long> loadVocab(Context context) throws Exception {
        byte[] bytes;
        try (java.io.InputStream input = context.getAssets().open("kokoro_vi/config.json")) {
            ByteArrayOutputStream out = new ByteArrayOutputStream();
            byte[] buffer = new byte[8192];
            int read;
            while ((read = input.read(buffer)) > 0) out.write(buffer, 0, read);
            bytes = out.toByteArray();
        }
        JSONObject vocabJson = new JSONObject(new String(bytes, StandardCharsets.UTF_8)).getJSONObject("vocab");
        Map<Character, Long> result = new HashMap<>();
        java.util.Iterator<String> keys = vocabJson.keys();
        while (keys.hasNext()) {
            String key = keys.next();
            if (key.length() == 1) result.put(key.charAt(0), vocabJson.getLong(key));
        }
        return result;
    }

    private float[] synthesizeChunk(String text, VietnameseKokoroVoice voice, float speed, String purpose) throws Exception {
        long totalStart = System.nanoTime();
        long stageStart = System.nanoTime();
        String phonemes = phonemize(text);
        long g2pMs = elapsedMs(stageStart);
        if (phonemes.trim().isEmpty()) return new float[0];

        stageStart = System.nanoTime();
        long[] ids = tokenIds(phonemes);
        long tokenizeMs = elapsedMs(stageStart);
        if (ids.length <= 2) return new float[0];

        stageStart = System.nanoTime();
        float[] style = selectStyle(voice, phonemes.length());
        long styleMs = elapsedMs(stageStart);
        float safeSpeed = Math.max(0.70f, Math.min(1.50f, speed));

        long nativeStart = System.nanoTime();
        float[] audio = nativeBridge.synthesize(ids, style, safeSpeed);
        long nativeCallMs = elapsedMs(nativeStart);
        long[] nativeTiming = nativeBridge.getLastInferenceTimingMicros();
        String nativeProfile = "";
        if (nativeTiming != null && nativeTiming.length >= 6) {
            nativeProfile = ", lockWaitUs=" + nativeTiming[0]
                    + ", inputPrepUs=" + nativeTiming[1]
                    + ", ortRunUs=" + nativeTiming[2]
                    + ", outputCopyUs=" + nativeTiming[3]
                    + ", nativeTotalUs=" + nativeTiming[4]
                    + ", cpuThreads=" + threadLabel((int) nativeTiming[5]);
        }
        if (appContext != null) {
            TtsDiagnostics.info(appContext, "profile", "chunk",
                    "purpose=" + purpose + ", textChars=" + text.length()
                            + ", phonemeChars=" + phonemes.length() + ", tokenIds=" + ids.length
                            + ", audioSamples=" + (audio == null ? 0 : audio.length)
                            + ", g2pMs=" + g2pMs + ", tokenizeMs=" + tokenizeMs
                            + ", styleMs=" + styleMs + ", nativeCallMs=" + nativeCallMs
                            + ", totalMs=" + elapsedMs(totalStart) + nativeProfile);
        }
        return audio;
    }

    private long[] tokenIds(String phonemes) {
        ArrayList<Long> idList = new ArrayList<>();
        for (int i = 0; i < phonemes.length(); i++) {
            Long id = vocab.get(phonemes.charAt(i));
            if (id != null) idList.add(id);
        }
        long[] ids = new long[idList.size() + 2];
        for (int i = 0; i < idList.size(); i++) ids[i + 1] = idList.get(i);
        return ids;
    }

    private synchronized float[] selectStyle(VietnameseKokoroVoice voice, int phonemeCount) throws Exception {
        if (voice != VietnameseKokoroVoice.DIEM_TRINH) {
            throw new IllegalArgumentException("Only Diễm Trinh is enabled in Stage 1.");
        }
        if (diemTrinhStyles == null) {
            byte[] bytes = java.nio.file.Files.readAllBytes(assets.voicepack.toPath());
            int expected = MAX_PHONEMES * STYLE_SIZE * Float.BYTES;
            if (bytes.length != expected) throw new IllegalStateException("Invalid voicepack size: " + bytes.length);
            diemTrinhStyles = new float[bytes.length / Float.BYTES];
            ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer().get(diemTrinhStyles);
            if (appContext != null) {
                TtsDiagnostics.info(appContext, "engine", "voicepack_loaded",
                        "voice=" + voice.id + ", bytes=" + bytes.length + ", floats=" + diemTrinhStyles.length);
            }
        }
        int row = Math.max(1, Math.min(MAX_PHONEMES, phonemeCount)) - 1;
        int start = row * STYLE_SIZE;
        float[] style = new float[STYLE_SIZE];
        System.arraycopy(diemTrinhStyles, start, style, 0, STYLE_SIZE);
        return style;
    }

    private List<String> splitWithBoundary(String text) {
        String normalized = WHITESPACE.matcher(text.trim()).replaceAll(" ");
        ArrayList<String> chunks = new ArrayList<>();
        if (normalized.isEmpty()) return chunks;
        Matcher matcher = SENTENCE_BOUNDARY.matcher(normalized);
        int start = 0;
        while (matcher.find()) {
            int end = matcher.end();
            if (end < normalized.length() && !Character.isWhitespace(normalized.charAt(end))) continue;
            String part = normalized.substring(start, end).trim();
            if (!part.isEmpty()) chunks.add(part);
            start = end;
        }
        String tail = normalized.substring(start).trim();
        if (!tail.isEmpty()) chunks.add(tail);
        return chunks;
    }

    private List<String> splitLongChunk(String chunk) throws Exception {
        ArrayList<String> single = new ArrayList<>();
        if (phonemize(chunk).length() <= MAX_PHONEMES) {
            single.add(chunk);
            return single;
        }
        ArrayList<String> result = new ArrayList<>();
        StringBuilder current = new StringBuilder();
        for (String word : WHITESPACE.split(chunk)) {
            String candidate = current.length() == 0 ? word : current + " " + word;
            if (phonemize(candidate).length() > MAX_PHONEMES && current.length() > 0) {
                result.add(current.toString());
                current = new StringBuilder(word);
            } else {
                current = new StringBuilder(candidate);
            }
        }
        if (current.length() > 0) result.add(current.toString());
        return result;
    }

    private String phonemize(String text) throws Exception {
        String normalized = text.replace('’', '\'').replace('‘', '\'');
        Matcher matcher = TOKEN_PATTERN.matcher(normalized);
        StringBuilder output = new StringBuilder();
        while (matcher.find()) {
            String token = matcher.group();
            if (token.trim().isEmpty()) output.append(' ');
            else if (WORD_TOKEN.matcher(token).matches()) {
                output.append(fixPhonemes(nativeBridge.phonemize(g2pHandle, token), token));
            } else output.append(fixPhonemes(token, null));
        }
        return output.toString().trim();
    }

    private String fixPhonemes(String raw, String sourceText) {
        if (raw == null) return "";
        String phonemes = raw
                .replace("tʃ", "ʧ").replace("t̪", "\uE100").replace("\uE100", "t")
                .replace("e-", "æ").replace("1", "→").replace("7", "→")
                .replace("2", "↘").replace("ɜ", "↗").replace("3", "↗")
                .replace("4", "↓").replace("5", "ʔ↗").replace("6", "ʔ↓")
                .replace("ɗ", "d").replace("ʐ", "ʒ").replace("̪", "")
                .replace("-", "").replace("–", "—").replace("*", "")
                .replace("/", " ").replace("&", " ").replace("'", "")
                .replace("’", "").replace("‘", "").replace("đ", "d").replace("̩", "");
        String source = sourceText == null ? "" : sourceText.toLowerCase(java.util.Locale.ROOT);
        if (source.startsWith("th")) phonemes = replaceFirstLiteral(phonemes, "t", "θ");
        else if (source.startsWith("tr")) phonemes = replaceFirstLiteral(phonemes, "ʧ", "ʈʂ");
        else if (source.startsWith("s") && !startsWithAny(source, "sc", "sh", "sk", "sl", "sm", "sn", "sp", "st", "sw")) {
            phonemes = replaceFirstLiteral(phonemes, "s", "ʂ");
        } else if (source.startsWith("gi") || GI_ACCENT.matcher(source).find()) {
            phonemes = replaceFirstLiteral(phonemes, "z", "ʝ");
        }
        return phonemes;
    }

    private static boolean startsWithAny(String value, String... prefixes) {
        for (String prefix : prefixes) if (value.startsWith(prefix)) return true;
        return false;
    }

    private static String replaceFirstLiteral(String value, String target, String replacement) {
        int index = value.indexOf(target);
        if (index < 0) return value;
        return value.substring(0, index) + replacement + value.substring(index + target.length());
    }

    private static byte[] toPcm16(float[] samples) {
        byte[] pcm = new byte[samples.length * 2];
        for (int i = 0; i < samples.length; i++) {
            float sample = Math.max(-1.0f, Math.min(1.0f, samples[i]));
            short value = (short) Math.round(sample * 32767.0f);
            pcm[i * 2] = (byte) (value & 0xff);
            pcm[i * 2 + 1] = (byte) ((value >>> 8) & 0xff);
        }
        return pcm;
    }

    private static long elapsedMs(long startedNanos) {
        return (System.nanoTime() - startedNanos) / 1_000_000L;
    }

    private static String threadLabel(int threads) {
        return threads == 0 ? "default" : Integer.toString(threads);
    }
}
