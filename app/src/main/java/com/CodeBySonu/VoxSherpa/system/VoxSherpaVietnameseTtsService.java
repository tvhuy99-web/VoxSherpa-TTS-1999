package com.CodeBySonu.VoxSherpa.system;

import android.media.AudioFormat;
import android.speech.tts.SynthesisCallback;
import android.speech.tts.SynthesisRequest;
import android.speech.tts.TextToSpeech;
import android.speech.tts.Voice;

import com.CodeBySonu.VoxSherpa.vietnamese.VietnameseKokoroEngine;
import com.CodeBySonu.VoxSherpa.vietnamese.VietnameseKokoroVoice;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;

/** Adds the bundled Vietnamese Kokoro backend without changing existing VoxSherpa engine paths. */
public class VoxSherpaVietnameseTtsService extends VoxSherpaTtsService {
    private volatile boolean vietnameseCancelled = false;

    @Override
    protected int onIsLanguageAvailable(String lang, String country, String variant) {
        if (isVietnameseLanguage(lang) && VietnameseKokoroEngine.isBundled(this)) {
            return TextToSpeech.LANG_COUNTRY_AVAILABLE;
        }
        return super.onIsLanguageAvailable(lang, country, variant);
    }

    @Override
    public String onGetDefaultVoiceNameFor(String lang, String country, String variant) {
        if (isVietnameseLanguage(lang) && VietnameseKokoroEngine.isBundled(this)) {
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
        } catch (Throwable ignored) {}

        if (VietnameseKokoroEngine.isBundled(this)) {
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
        }
        return result;
    }

    @Override
    public int onIsValidVoiceName(String voiceName) {
        if (VietnameseKokoroEngine.isBundled(this)
                && VietnameseKokoroVoice.fromAndroidVoiceName(voiceName) != null) {
            return TextToSpeech.SUCCESS;
        }
        return super.onIsValidVoiceName(voiceName);
    }

    @Override
    public int onLoadVoice(String voiceName) {
        if (VietnameseKokoroEngine.isBundled(this)
                && VietnameseKokoroVoice.fromAndroidVoiceName(voiceName) != null) {
            return TextToSpeech.SUCCESS;
        }
        return super.onLoadVoice(voiceName);
    }

    @Override
    protected void onStop() {
        vietnameseCancelled = true;
        try { VietnameseKokoroEngine.getInstance().cancel(); } catch (Throwable ignored) {}
        super.onStop();
    }

    @Override
    public void onDestroy() {
        vietnameseCancelled = true;
        try { VietnameseKokoroEngine.getInstance().release(); } catch (Throwable ignored) {}
        super.onDestroy();
    }

    @Override
    protected void onSynthesizeText(SynthesisRequest request, SynthesisCallback callback) {
        if (!shouldUseVietnamese(request)) {
            super.onSynthesizeText(request, callback);
            return;
        }

        vietnameseCancelled = false;
        CharSequence chars = request.getCharSequenceText();
        String text = chars == null ? "" : chars.toString().trim();
        if (text.isEmpty()) {
            if (callback.start(VietnameseKokoroEngine.SAMPLE_RATE, AudioFormat.ENCODING_PCM_16BIT, 1)
                    == TextToSpeech.SUCCESS) {
                callback.done();
            } else {
                callback.error();
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
            return;
        }

        final float speed = request.getSpeechRate() > 0
                ? request.getSpeechRate() / 100.0f
                : 1.0f;
        final int maxBuffer = Math.max(1024, callback.getMaxBufferSize());
        final boolean[] writeFailed = {false};

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
                            return false;
                        }
                    }
                    return !vietnameseCancelled;
                }
        );

        if (vietnameseCancelled) return;
        if (!emitted || writeFailed[0]) {
            callback.error();
            return;
        }
        callback.done();
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
