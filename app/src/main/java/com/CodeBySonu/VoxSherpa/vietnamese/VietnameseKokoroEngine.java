package com.CodeBySonu.VoxSherpa.vietnamese;

import android.content.Context;
import android.content.SharedPreferences;

import com.CodeBySonu.VoxSherpa.Sonic;
import com.CodeBySonu.VoxSherpa.system.TtsDiagnostics;

import org.json.JSONObject;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
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
    private static final int MAX_STYLE_CACHE = 3;
    private static final String PERF_PREFS = "kokoro_vi_perf";
    private static final String APP_PREFS = "sp3";
    private static final String PREF_CPU_THREADS = "cpu_threads";
    private static final String PREF_NNAPI_ENABLED = "nnapi_enabled";
    private static final Pattern SENTENCE_BOUNDARY = Pattern.compile("[.!?…]+(?:[\\\"”’)]*)");
    private static final Pattern EFFECT_TOKEN = Pattern.compile("(\\[[a-zA-Z]+\\]|\\.\\.\\.|[.,!?।])");
    private static final Pattern WORD_TOKEN = Pattern.compile("^[A-Za-zÀ-ỹĐđ]+(?:[-'][A-Za-zÀ-ỹĐđ]+)*$");
    private static final Pattern TOKEN_PATTERN = Pattern.compile("[A-Za-zÀ-ỹĐđ]+(?:[-'][A-Za-zÀ-ỹĐđ]+)*|\\s+|.");
    private static final Pattern WHITESPACE = Pattern.compile("\\s+");
    private static final Pattern GI_ACCENT = Pattern.compile("^g[iìíỉĩị]");
    private static final Random RANDOM = new Random();
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
    private volatile boolean activeNnapi;
    private volatile int activeCpuThreads = 4;
    private Map<Character, Long> vocab = new HashMap<>();
    private final LinkedHashMap<String, float[]> voiceStyleCache =
            new LinkedHashMap<String, float[]>(4, 0.75f, true) {
                @Override
                protected boolean removeEldestEntry(Map.Entry<String, float[]> eldest) {
                    boolean remove = size() > MAX_STYLE_CACHE;
                    if (remove && appContext != null) {
                        TtsDiagnostics.info(appContext, "engine", "voicepack_cache_evict",
                                "voice=" + eldest.getKey() + ", remaining=" + MAX_STYLE_CACHE);
                    }
                    return remove;
                }
            };
    private Context appContext;

    public interface PcmConsumer { boolean onPcm(byte[] pcm); }

    private static final class AudioSettings {
        float speed;
        float pitch;
        boolean punctuation;
        boolean emotion;
        float silence;
        String source;
    }

    private static final class EffectProfile {
        float speed;
        float pitch;
        float volume;

        EffectProfile(float speed, float pitch, float volume) {
            this.speed = speed;
            this.pitch = pitch;
            this.volume = volume;
        }
    }

    private static final class StreamState {
        boolean emitted;
        boolean firstPcm = true;
        int chunks;
    }

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

    public boolean isNnapiRequested(Context context) {
        return context != null && context.getSharedPreferences(PERF_PREFS, Context.MODE_PRIVATE)
                .getBoolean(PREF_NNAPI_ENABLED, false);
    }

    public boolean isNnapiActive() {
        return activeNnapi;
    }

    public String performanceState(Context context) {
        SharedPreferences prefs = context.getSharedPreferences(PERF_PREFS, Context.MODE_PRIVATE);
        int saved = prefs.getInt(PREF_CPU_THREADS, 4);
        boolean nnapiRequested = prefs.getBoolean(PREF_NNAPI_ENABLED, false);
        synchronized (this) {
            return "ready=" + isReady() + ", modelReady=" + modelReady + ", warm=" + modelWarm
                    + ", g2pReady=" + (g2pHandle != 0L) + ", provider=" + (activeNnapi ? "NNAPI" : "CPU")
                    + ", nnapiRequested=" + nnapiRequested + ", activeCpuThreads=" + activeCpuThreads
                    + ", savedCpuThreads=" + saved + " (0=ORT default)"
                    + ", cachedVoiceStyles=" + voiceStyleCache.size();
        }
    }

    public void prewarmAsync(Context context, String reason) {
        if (context == null || !isBundled(context)) return;
        final Context app = context.getApplicationContext();
        if (isReady()) {
            TtsDiagnostics.info(app, "prewarm", "already_hot", "reason=" + reason + ", " + performanceState(app));
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
                        "reason=" + reason + ", elapsedMs=" + elapsedMs(started) + ", " + performanceState(app));
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
        float[] audio = synthesizeChunk("xin", VietnameseKokoroVoice.DEFAULT, 1.0f, "prewarm");
        modelWarm = audio != null && audio.length > 0;
        TtsDiagnostics.info(context, "prewarm", "tiny_inference",
                "reason=" + reason + ", voice=" + VietnameseKokoroVoice.DEFAULT.id
                        + ", audioSamples=" + (audio == null ? 0 : audio.length)
                        + ", elapsedMs=" + elapsedMs(started));
    }

    public void cancel() {
        long newGeneration = generation.incrementAndGet();
        boolean nativeTerminated = false;
        try {
            if (VietnameseKokoroNative.isAvailable()) nativeTerminated = nativeBridge.cancelActiveRun();
        } catch (Throwable t) {
            if (appContext != null) TtsDiagnostics.error(appContext, "engine", "native_cancel_failed", t.toString(), t);
        }
        if (appContext != null) {
            TtsDiagnostics.info(appContext, "engine", "cancel",
                    "generation=" + newGeneration + ", nativeRunTerminated=" + nativeTerminated
                            + ", modelReady=" + modelReady + ", modelWarm=" + modelWarm
                            + ", g2pReady=" + (g2pHandle != 0L));
        }
    }

    public synchronized boolean setNnapiEnabled(Context context, boolean enabled) throws Exception {
        Context app = context.getApplicationContext();
        appContext = app;
        SharedPreferences prefs = app.getSharedPreferences(PERF_PREFS, Context.MODE_PRIVATE);
        prefs.edit().putBoolean(PREF_NNAPI_ENABLED, enabled).apply();
        long started = System.nanoTime();
        cancel();
        if (modelReady) nativeBridge.destroyEngine();
        modelReady = false;
        modelWarm = false;
        activeNnapi = false;

        prepare(app);
        if (!modelWarm) {
            float[] audio = synthesizeChunk("xin", VietnameseKokoroVoice.DEFAULT, 1.0f, "provider_switch");
            modelWarm = audio != null && audio.length > 0;
        }

        boolean active = activeNnapi;
        if (enabled && !active) {
            prefs.edit().putBoolean(PREF_NNAPI_ENABLED, false).apply();
            TtsDiagnostics.warn(app, "provider", "nnapi_fallback",
                    "NNAPI was requested but the native session fell back to CPU; NNAPI preference was disabled.");
        }
        TtsDiagnostics.info(app, "provider", "configured",
                "requested=" + enabled + ", active=" + active + ", elapsedMs=" + elapsedMs(started)
                        + ", " + performanceState(app));
        return active;
    }

    public boolean synthesizeStreaming(Context context, String text, VietnameseKokoroVoice voice,
                                       float speed, PcmConsumer consumer) {
        if (context == null || text == null || text.trim().isEmpty() || consumer == null) return false;
        if (voice == null) voice = VietnameseKokoroVoice.DEFAULT;
        final VietnameseKokoroVoice selectedVoice = voice;
        final long requestGeneration = generation.incrementAndGet();
        final long startedAt = System.nanoTime();
        final AudioSettings settings = resolveAudioSettings(context, speed);

        TtsDiagnostics.info(context, "settings", "vietnamese_audio_settings",
                "source=" + settings.source + ", effectiveSpeed=" + settings.speed
                        + ", effectivePitch=" + settings.pitch + ", smartPunct=" + settings.punctuation
                        + ", emotionTags=" + settings.emotion + ", silenceScale=" + settings.silence);

        try {
            long prepareStart = System.nanoTime();
            prepare(context.getApplicationContext());
            long prepareMs = elapsedMs(prepareStart);
            StreamState state = new StreamState();

            if (settings.punctuation || settings.emotion) {
                streamWithEffects(context, text, selectedVoice, settings, requestGeneration,
                        startedAt, prepareMs, state, consumer);
            } else {
                for (String chunk : splitWithBoundary(text)) {
                    if (requestGeneration != generation.get()) return state.emitted;
                    emitText(context, chunk, selectedVoice, settings.speed, settings.pitch, 1.0f,
                            requestGeneration, startedAt, prepareMs, state, consumer);
                }
            }

            TtsDiagnostics.info(context, "engine", "stream_done",
                    "generation=" + requestGeneration + ", voice=" + selectedVoice.id
                            + ", chunks=" + state.chunks + ", emitted=" + state.emitted
                            + ", elapsedMs=" + elapsedMs(startedAt));
            return state.emitted;
        } catch (Throwable t) {
            if (requestGeneration != generation.get()) {
                TtsDiagnostics.info(context, "engine", "stream_cancelled",
                        "generation=" + requestGeneration + ", voice=" + selectedVoice.id
                                + ", elapsedMs=" + elapsedMs(startedAt));
                return false;
            }
            TtsDiagnostics.error(context, "engine", "synthesis_exception",
                    "generation=" + requestGeneration + ", voice=" + selectedVoice.id
                            + ", speed=" + settings.speed + ", pitch=" + settings.pitch
                            + ", chars=" + text.length(), t);
            return false;
        }
    }

    private AudioSettings resolveAudioSettings(Context context, float callerSpeed) {
        SharedPreferences sp3 = context.getSharedPreferences(APP_PREFS, Context.MODE_PRIVATE);
        float appSpeed = clamp(sp3.getFloat("voice_speed", 1.0f), 0.25f, 2.0f);
        float appPitch = clamp(sp3.getFloat("voice_pitch", 1.0f), 0.25f, 2.0f);
        boolean systemTts = context instanceof android.speech.tts.TextToSpeechService;

        AudioSettings out = new AudioSettings();
        out.source = systemTts ? "system_tts" : "generate";
        out.speed = systemTts
                ? clamp((callerSpeed > 0f ? callerSpeed : 1.0f) * appSpeed, 0.25f, 2.0f)
                : clamp(callerSpeed > 0f ? callerSpeed : appSpeed, 0.25f, 2.0f);
        out.pitch = appPitch;
        out.punctuation = sp3.getBoolean("smart_punct", false);
        out.emotion = sp3.getBoolean("emotion_tags", false);
        out.silence = clamp(sp3.getFloat("silence_scale", 0.2f), 0.0f, 1.0f);
        return out;
    }

    private void streamWithEffects(Context context, String text, VietnameseKokoroVoice voice,
                                   AudioSettings settings, long requestGeneration, long startedAt,
                                   long prepareMs, StreamState state, PcmConsumer consumer) throws Exception {
        EffectProfile profile = new EffectProfile(settings.speed, settings.pitch, 1.0f);
        Matcher matcher = EFFECT_TOKEN.matcher(text);
        int lastEnd = 0;

        while (matcher.find()) {
            if (requestGeneration != generation.get()) return;
            String part = text.substring(lastEnd, matcher.start()).trim();
            if (!part.isEmpty()) {
                emitText(context, part, voice, profile.speed, profile.pitch, profile.volume,
                        requestGeneration, startedAt, prepareMs, state, consumer);
                if (requestGeneration != generation.get()) return;
            }

            String token = matcher.group();
            if (token.startsWith("[")) {
                if (settings.emotion) profile = emotionProfile(token, settings.speed, settings.pitch);
            } else if (settings.punctuation && state.emitted) {
                int silenceMs = punctuationSilenceMs(token, settings.silence, profile.speed);
                if (silenceMs > 0) {
                    byte[] silence = new byte[(SAMPLE_RATE * 2 * silenceMs) / 1000];
                    if ((silence.length & 1) != 0) silence = new byte[silence.length + 1];
                    if (!consumer.onPcm(silence)) return;
                }
            }
            lastEnd = matcher.end();
        }

        String tail = text.substring(lastEnd).trim();
        if (!tail.isEmpty() && requestGeneration == generation.get()) {
            emitText(context, tail, voice, profile.speed, profile.pitch, profile.volume,
                    requestGeneration, startedAt, prepareMs, state, consumer);
        }
    }

    private EffectProfile emotionProfile(String token, float baseSpeed, float basePitch) {
        String tag = token == null ? "" : token.toLowerCase(java.util.Locale.ROOT);
        switch (tag) {
            case "[whispers]":
            case "[whisper]":
                return new EffectProfile(clamp(baseSpeed * 0.95f, 0.25f, 2.0f),
                        clamp(basePitch * 1.05f, 0.25f, 2.0f), 0.65f);
            case "[angry]":
                return new EffectProfile(clamp(baseSpeed * 1.05f, 0.25f, 2.0f),
                        clamp(basePitch * 0.95f, 0.25f, 2.0f), 1.15f);
            case "[sad]":
                return new EffectProfile(clamp(baseSpeed * 0.92f, 0.25f, 2.0f),
                        clamp(basePitch * 0.98f, 0.25f, 2.0f), 0.80f);
            case "[sarcastically]":
            case "[sarcastic]":
                return new EffectProfile(clamp(baseSpeed * 1.02f, 0.25f, 2.0f),
                        clamp(basePitch * 0.95f, 0.25f, 2.0f), 1.0f);
            case "[giggles]":
            case "[giggle]":
                return new EffectProfile(clamp(baseSpeed * 1.05f, 0.25f, 2.0f),
                        clamp(basePitch * 1.10f, 0.25f, 2.0f), 1.10f);
            case "[normal]":
            default:
                return new EffectProfile(baseSpeed, basePitch, 1.0f);
        }
    }

    private int punctuationSilenceMs(String token, float silenceScale, float currentSpeed) {
        int base;
        switch (token) {
            case ",": base = 150; break;
            case "!": base = 200; break;
            case "?": base = 250; break;
            case ".":
            case "।": base = 300; break;
            case "...": base = 450; break;
            default: return 0;
        }
        float multiplier = silenceScale * 2.0f;
        int adjusted = (int) ((base * multiplier) / Math.max(0.25f, currentSpeed));
        int jitter = (int) (adjusted * 0.10f);
        if (jitter > 0) adjusted += RANDOM.nextInt(jitter * 2 + 1) - jitter;
        return Math.max(0, adjusted);
    }

    private void emitText(Context context, String text, VietnameseKokoroVoice voice,
                          float speed, float pitch, float volume, long requestGeneration,
                          long startedAt, long prepareMs, StreamState state,
                          PcmConsumer consumer) throws Exception {
        long splitStart = System.nanoTime();
        List<String> safeChunks = splitLongChunk(text);
        long splitMs = elapsedMs(splitStart);
        for (String safeChunk : safeChunks) {
            if (requestGeneration != generation.get()) return;
            state.chunks++;
            float[] audio = synthesizeChunk(safeChunk, voice, speed, "request");
            if (requestGeneration != generation.get()) return;
            if (audio == null || audio.length == 0) continue;

            long pcmStart = System.nanoTime();
            byte[] pcm = toPcm16(audio);
            pcm = applyPitch(pcm, pitch);
            if (Math.abs(volume - 1.0f) > 0.001f) applyVolumeInPlace(pcm, volume);
            long pcmMs = elapsedMs(pcmStart);

            if (state.firstPcm) {
                state.firstPcm = false;
                TtsDiagnostics.info(context, "latency", "first_pcm_ready",
                        "generation=" + requestGeneration + ", voice=" + voice.id
                                + ", requestToPcmMs=" + elapsedMs(startedAt)
                                + ", prepareMs=" + prepareMs + ", splitCheckMs=" + splitMs
                                + ", pcmConvertAndPitchMs=" + pcmMs + ", speed=" + speed
                                + ", pitch=" + pitch + ", bytes=" + pcm.length);
            }
            state.emitted = true;
            if (!consumer.onPcm(pcm)) return;
        }
    }

    private byte[] applyPitch(byte[] pcm, float pitch) {
        float safePitch = clamp(pitch, 0.25f, 2.0f);
        if (pcm == null || pcm.length < 2 || Math.abs(safePitch - 1.0f) < 0.001f) return pcm;
        try {
            int samples = pcm.length / 2;
            short[] in = new short[samples];
            for (int i = 0; i < samples; i++) {
                int lo = pcm[i * 2] & 0xff;
                int hi = pcm[i * 2 + 1] << 8;
                in[i] = (short) (lo | hi);
            }
            Sonic sonic = new Sonic(SAMPLE_RATE, 1);
            sonic.setPitch(safePitch);
            sonic.writeShortToStream(in, in.length);
            sonic.flushStream();
            int available = sonic.samplesAvailable();
            if (available <= 0) return pcm;
            short[] out = new short[available];
            int read = sonic.readShortFromStream(out, available);
            if (read <= 0) return pcm;
            byte[] result = new byte[read * 2];
            for (int i = 0; i < read; i++) {
                result[i * 2] = (byte) (out[i] & 0xff);
                result[i * 2 + 1] = (byte) ((out[i] >>> 8) & 0xff);
            }
            return result;
        } catch (Throwable t) {
            if (appContext != null) TtsDiagnostics.warn(appContext, "settings", "pitch_fallback",
                    "Sonic pitch processing failed; using unmodified PCM. pitch=" + safePitch + ", error=" + t);
            return pcm;
        }
    }

    private static void applyVolumeInPlace(byte[] pcm, float volume) {
        if (pcm == null) return;
        for (int i = 0; i + 1 < pcm.length; i += 2) {
            int lo = pcm[i] & 0xff;
            int hi = pcm[i + 1] << 8;
            int sample = (short) (lo | hi);
            int scaled = Math.round(sample * volume);
            scaled = Math.max(-32768, Math.min(32767, scaled));
            pcm[i] = (byte) (scaled & 0xff);
            pcm[i + 1] = (byte) ((scaled >>> 8) & 0xff);
        }
    }

    public void retainAcrossServiceDestroy(Context context) {
        if (context != null) {
            TtsDiagnostics.info(context, "engine", "session_retained",
                    "TTS service destroyed; keeping Vietnamese ONNX/G2P session hot in process. "
                            + performanceState(context));
        }
    }

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
        cancel();
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
        activeNnapi = false;
        assets = null;
        vocab.clear();
        voiceStyleCache.clear();
        if (appContext != null) TtsDiagnostics.info(appContext, "engine", "released", "Vietnamese engine resources released.");
    }

    public synchronized String benchmarkCpuThreads(Context context) throws Exception {
        Context app = context.getApplicationContext();
        appContext = app;
        long totalStart = System.nanoTime();
        SharedPreferences prefs = app.getSharedPreferences(PERF_PREFS, Context.MODE_PRIVATE);
        boolean restoreNnapi = prefs.getBoolean(PREF_NNAPI_ENABLED, false);
        TtsDiagnostics.info(app, "benchmark", "cpu_start",
                "Testing ORT default/3/4/5/6 CPU threads with 2 warmups + 5 measured runs each; TTS requests may block during this benchmark.");
        prepare(app);
        String phonemes = phonemize("xin chào");
        long[] ids = tokenIds(phonemes);
        float[] style = selectStyle(VietnameseKokoroVoice.DEFAULT, phonemes.length());
        String json = nativeBridge.benchmarkCpuThreads(assets.model.getAbsolutePath(), ids, style, 1.0f, 2, 5);
        JSONObject result = new JSONObject(json);
        int best = result.getInt("bestThreads");
        prefs.edit().putInt(PREF_CPU_THREADS, best).apply();
        activeCpuThreads = best;
        modelReady = true;
        modelWarm = true;
        activeNnapi = false;

        if (restoreNnapi) {
            nativeBridge.destroyEngine();
            modelReady = false;
            modelWarm = false;
            prepare(app);
            if (!modelWarm) {
                float[] warm = synthesizeChunk("xin", VietnameseKokoroVoice.DEFAULT, 1.0f, "benchmark_restore");
                modelWarm = warm != null && warm.length > 0;
            }
        }

        TtsDiagnostics.info(app, "benchmark", "cpu_complete",
                "elapsedMs=" + elapsedMs(totalStart) + ", result=" + json
                        + ", selected=" + threadLabel(best) + ", providerAfterBenchmark="
                        + (activeNnapi ? "NNAPI" : "CPU"));
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
            activeCpuThreads = prefs.getInt(PREF_CPU_THREADS, 4);
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
        float safeSpeed = clamp(speed, 0.25f, 2.0f);

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
                    "purpose=" + purpose + ", voice=" + voice.id + ", provider="
                            + (activeNnapi ? "NNAPI" : "CPU") + ", textChars=" + text.length()
                            + ", phonemeChars=" + phonemes.length() + ", tokenIds=" + ids.length
                            + ", audioSamples=" + (audio == null ? 0 : audio.length)
                            + ", g2pMs=" + g2pMs + ", tokenizeMs=" + tokenizeMs
                            + ", styleMs=" + styleMs + ", nativeCallMs=" + nativeCallMs
                            + ", requestedSpeed=" + speed + ", effectiveModelSpeed=" + safeSpeed
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
        if (voice == null) voice = VietnameseKokoroVoice.DEFAULT;
        float[] styles = voiceStyleCache.get(voice.id);
        if (styles == null) {
            if (appContext == null) throw new IllegalStateException("Vietnamese engine context is unavailable.");
            File voiceFile = VietnameseKokoroAssetStore.ensureVoice(appContext, voice);
            byte[] bytes = java.nio.file.Files.readAllBytes(voiceFile.toPath());
            int expected = MAX_PHONEMES * STYLE_SIZE * Float.BYTES;
            if (bytes.length != expected) {
                throw new IllegalStateException("Invalid voicepack size for " + voice.id + ": " + bytes.length);
            }
            styles = new float[bytes.length / Float.BYTES];
            ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer().get(styles);
            voiceStyleCache.put(voice.id, styles);
            TtsDiagnostics.info(appContext, "engine", "voicepack_loaded",
                    "voice=" + voice.id + ", displayName=" + voice.displayName
                            + ", bytes=" + bytes.length + ", floats=" + styles.length
                            + ", cacheSize=" + voiceStyleCache.size());
        }
        int row = Math.max(1, Math.min(MAX_PHONEMES, phonemeCount)) - 1;
        int start = row * STYLE_SIZE;
        float[] style = new float[STYLE_SIZE];
        System.arraycopy(styles, start, style, 0, STYLE_SIZE);
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

    private static float clamp(float value, float min, float max) {
        return Math.max(min, Math.min(max, value));
    }

    private static long elapsedMs(long startedNanos) {
        return (System.nanoTime() - startedNanos) / 1_000_000L;
    }

    private static String threadLabel(int threads) {
        return threads == 0 ? "default" : Integer.toString(threads);
    }
}
