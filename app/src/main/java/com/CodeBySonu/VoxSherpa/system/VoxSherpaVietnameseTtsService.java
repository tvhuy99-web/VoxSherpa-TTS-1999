package com.CodeBySonu.VoxSherpa.system;

import android.media.AudioFormat;
import android.speech.tts.SynthesisCallback;
import android.speech.tts.SynthesisRequest;
import android.speech.tts.TextToSpeech;
import android.speech.tts.Voice;

import com.CodeBySonu.VoxSherpa.vietnamese.VietnameseKokoroEngine;
import com.CodeBySonu.VoxSherpa.vietnamese.VietnameseKokoroNative;
import com.CodeBySonu.VoxSherpa.vietnamese.VietnameseKokoroVoice;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.atomic.AtomicLong;

/** Adds the bundled Vietnamese Kokoro backend without changing existing VoxSherpa engine paths. */
public class VoxSherpaVietnameseTtsService extends VoxSherpaTtsService {
    private volatile boolean vietnameseCancelled = false;
    private final AtomicLong requestCounter = new AtomicLong(0L);

    @Override
    public void onCreate() {
        super.onCreate();
        boolean nativeAvailable = VietnameseKokoroNative.isAvailable();
        boolean bundled = VietnameseKokoroEngine.isBundled(this);
        TtsDiagnostics.info(this, "service", "created",
                "nativeAvailable=" + nativeAvailable + ", bundled=" + bundled
                        + ", loadError=" + VietnameseKokoroNative.loadError());
        if (!nativeAvailable) {
            TtsDiagnostics.error(this, "native", "load_failed",
                    "Vietnamese JNI backend unavailable: " + VietnameseKokoroNative.loadError(), null);
        }
    }

    @Override
    protected int onIsLanguageAvailable(String lang, String country, String variant) {
        boolean bundled = VietnameseKokoroEngine.isBundled(this);
        if (isVietnameseLanguage(lang) && bundled) {
            TtsDiagnostics.info(this, "service", "language_available",
                    "lang=" + lang + ", country=" + country + ", result=LANG_COUNTRY_AVAILABLE");
            return TextToSpeech.LANG_COUNTRY_AVAILABLE;
        }
        int result = super.onIsLanguageAvailable(lang, country, variant);
        if (isVietnameseLanguage(lang)) {
            TtsDiagnostics.warn(this, "service", "language_unavailable",
                    "lang=" + lang + ", nativeAvailable=" + VietnameseKokoroNative.isAvailable()
                            + ", bundled=" + bundled + ", nativeError=" + VietnameseKokoroNative.loadError()
                            + ", superResult=" + result);
        }
        return result;
    }

    @Override
    public String onGetDefaultVoiceNameFor(String lang, String country, String variant) {
        if (isVietnameseLanguage(lang) && VietnameseKokoroEngine.isBundled(this)) {
            TtsDiagnostics.info(this, "service", "default_voice",
                    "lang=" + lang + ", voice=" + VietnameseKokoroVoice.DIEM_TRINH.androidVoiceName);
            return VietnameseKokoroVoice.DIEM_TRINH.androidVoiceName;
        }
        return super.onGetDefaultVoiceNameFor(lang, country, variant);
    }

    @Override
    public List<Voice> onGetVoices() {
        List<Voice> result = new ArrayList<>();
        try {
            List<Voice> existing = super.onGetVoices();
            if (existing != null) result.addAll(existing);
        } catch (Throwable t) {
            TtsDiagnostics.error(this, "service", "base_voice_enumeration_failed", t.toString(), t);
        }

        boolean bundled = VietnameseKokoroEngine.isBundled(this);
        if (bundled) {
            boolean alreadyPresent = false;
            for (Voice voice : result) {
                if (VietnameseKokoroVoice.DIEM_TRINH.androidVoiceName.equals(voice.getName())) {
                    alreadyPresent = true;
                    break;
                }
            }
            if (!alreadyPresent) {
                result.add(new Voice(
                        VietnameseKokoroVoice.DIEM_TRINH.androidVoiceName,
                        VietnameseKokoroVoice.LOCALE,
                        Voice.QUALITY_VERY_HIGH,
                        Voice.LATENCY_NORMAL,
                        false,
                        new HashSet<>()
                ));
            }
            TtsDiagnostics.info(this, "service", "voices_enumerated",
                    "total=" + result.size() + ", vietnameseAdded=" + !alreadyPresent
                            + ", voice=" + VietnameseKokoroVoice.DIEM_TRINH.androidVoiceName);
        } else {
            TtsDiagnostics.warn(this, "service", "vietnamese_voice_hidden",
                    "nativeAvailable=" + VietnameseKokoroNative.isAvailable()
                            + ", nativeError=" + VietnameseKokoroNative.loadError());
        }
        return result;
    }

    @Override
    public int onIsValidVoiceName(String voiceName) {
        if (VietnameseKokoroEngine.isBundled(this)
                && VietnameseKokoroVoice.fromAndroidVoiceName(voiceName) != null) {
            TtsDiagnostics.info(this, "service", "voice_valid", "voice=" + voiceName);
            return TextToSpeech.SUCCESS;
        }
        return super.onIsValidVoiceName(voiceName);
    }

    @Override
    public int onLoadVoice(String voiceName) {
        if (VietnameseKokoroEngine.isBundled(this)
                && VietnameseKokoroVoice.fromAndroidVoiceName(voiceName) != null) {
            TtsDiagnostics.info(this, "service", "voice_loaded", "voice=" + voiceName);
            return TextToSpeech.SUCCESS;
        }
        if (voiceName != null && voiceName.contains("VoxSherpa_vi_")) {
            TtsDiagnostics.warn(this, "service", "voice_load_rejected",
                    "voice=" + voiceName + ", nativeAvailable=" + VietnameseKokoroNative.isAvailable()
                            + ", nativeError=" + VietnameseKokoroNative.loadError());
        }
        return super.onLoadVoice(voiceName);
    }

    @Override
    protected void onStop() {
        vietnameseCancelled = true;
        TtsDiagnostics.info(this, "service", "stop", "TTS stop/cancel requested.");
        try { VietnameseKokoroEngine.getInstance().cancel(); }
        catch (Throwable t) { TtsDiagnostics.error(this, "service", "cancel_failed", t.toString(), t); }
        super.onStop();
    }

    @Override
    public void onDestroy() {
        vietnameseCancelled = true;
        TtsDiagnostics.info(this, "service", "destroy", "TTS service is being destroyed.");
        try { VietnameseKokoroEngine.getInstance().release(); }
        catch (Throwable t) { TtsDiagnostics.error(this, "service", "release_failed", t.toString(), t); }
        super.onDestroy();
    }

    @Override
    protected void onSynthesizeText(SynthesisRequest request, SynthesisCallback callback) {
        final long requestId = requestCounter.incrementAndGet();
        final long startedAt = System.nanoTime();
        CharSequence chars = request.getCharSequenceText();
        String previewText = chars == null ? "" : chars.toString();
        if (previewText.length() > 120) previewText = previewText.substring(0, 120) + "…";
        boolean useVietnamese = shouldUseVietnamese(request);
        TtsDiagnostics.info(this, "synthesis", "request",
                "id=" + requestId + ", route=" + (useVietnamese ? "kokoro_vi" : "base")
                        + ", lang=" + request.getLanguage() + ", country=" + request.getCountry()
                        + ", voice=" + request.getVoiceName() + ", rate=" + request.getSpeechRate()
                        + ", chars=" + (chars == null ? 0 : chars.length()) + ", text=" + previewText);

        if (!useVietnamese) {
            super.onSynthesizeText(request, callback);
            return;
        }

        vietnameseCancelled = false;
        String text = chars == null ? "" : chars.toString().trim();
        if (text.isEmpty()) {
            if (callback.start(VietnameseKokoroEngine.SAMPLE_RATE, AudioFormat.ENCODING_PCM_16BIT, 1)
                    == TextToSpeech.SUCCESS) {
                callback.done();
                TtsDiagnostics.info(this, "synthesis", "empty_done", "id=" + requestId);
            } else {
                callback.error();
                TtsDiagnostics.warn(this, "synthesis", "callback_start_failed", "id=" + requestId + ", emptyText=true");
            }
            return;
        }

        int startStatus = callback.start(
                VietnameseKokoroEngine.SAMPLE_RATE,
                AudioFormat.ENCODING_PCM_16BIT,
                1
        );
        if (startStatus != TextToSpeech.SUCCESS) {
            callback.error();
            TtsDiagnostics.warn(this, "synthesis", "callback_start_failed",
                    "id=" + requestId + ", status=" + startStatus);
            return;
        }

        final float speed = request.getSpeechRate() > 0
                ? request.getSpeechRate() / 100.0f
                : 1.0f;
        final int maxBuffer = Math.max(1024, callback.getMaxBufferSize());
        final boolean[] writeFailed = {false};
        final long[] pcmBytes = {0L};

        boolean emitted = VietnameseKokoroEngine.getInstance().synthesizeStreaming(
                this,
                text,
                VietnameseKokoroVoice.DIEM_TRINH,
                speed,
                pcm -> {
                    for (int offset = 0; offset < pcm.length; offset += maxBuffer) {
                        if (vietnameseCancelled) return false;
                        int count = Math.min(maxBuffer, pcm.length - offset);
                        if (callback.audioAvailable(pcm, offset, count) != TextToSpeech.SUCCESS) {
                            writeFailed[0] = true;
                            TtsDiagnostics.warn(this, "synthesis", "audio_write_failed",
                                    "id=" + requestId + ", offset=" + offset + ", count=" + count);
                            return false;
                        }
                        pcmBytes[0] += count;
                    }
                    return !vietnameseCancelled;
                }
        );

        long elapsedMs = (System.nanoTime() - startedAt) / 1_000_000L;
        if (vietnameseCancelled) {
            TtsDiagnostics.info(this, "synthesis", "cancelled",
                    "id=" + requestId + ", elapsedMs=" + elapsedMs + ", pcmBytes=" + pcmBytes[0]);
            return;
        }
        if (!emitted || writeFailed[0]) {
            callback.error();
            TtsDiagnostics.warn(this, "synthesis", "failed",
                    "id=" + requestId + ", emitted=" + emitted + ", writeFailed=" + writeFailed[0]
                            + ", elapsedMs=" + elapsedMs + ", pcmBytes=" + pcmBytes[0]);
            return;
        }
        callback.done();
        TtsDiagnostics.info(this, "synthesis", "done",
                "id=" + requestId + ", elapsedMs=" + elapsedMs + ", pcmBytes=" + pcmBytes[0]);
    }

    private boolean shouldUseVietnamese(SynthesisRequest request) {
        if (!VietnameseKokoroEngine.isBundled(this)) return false;
        String voiceName = request.getVoiceName();
        if (VietnameseKokoroVoice.fromAndroidVoiceName(voiceName) != null) return true;
        return isVietnameseLanguage(request.getLanguage());
    }

    private static boolean isVietnameseLanguage(String lang) {
        if (lang == null) return false;
        String value = lang.trim().toLowerCase(Locale.ROOT);
        return value.equals("vi") || value.equals("vie") || value.startsWith("vi-") || value.startsWith("vi_");
    }
}
