package com.CodeBySonu.VoxSherpa;

import android.app.Activity;
import android.content.Context;
import android.content.SharedPreferences;
import android.graphics.Bitmap;
import android.graphics.Color;
import android.media.AudioAttributes;
import android.media.AudioFormat;
import android.media.AudioManager;
import android.media.AudioTrack;
import android.os.Bundle;
import android.text.Editable;
import android.text.TextWatcher;
import android.view.MotionEvent;
import android.view.View;
import android.view.accessibility.AccessibilityNodeInfo;
import android.widget.ArrayAdapter;
import android.widget.EditText;
import android.widget.ImageView;
import android.widget.ProgressBar;
import android.widget.TextView;

import androidx.appcompat.widget.ListPopupWindow;
import androidx.fragment.app.Fragment;
import androidx.fragment.app.FragmentActivity;
import androidx.fragment.app.FragmentManager;

import com.CodeBySonu.VoxSherpa.system.TtsDiagnostics;
import com.CodeBySonu.VoxSherpa.vietnamese.VietnameseKokoroEngine;
import com.CodeBySonu.VoxSherpa.vietnamese.VietnameseKokoroVoice;
import com.google.android.material.card.MaterialCardView;
import com.google.android.material.snackbar.Snackbar;

import java.io.ByteArrayOutputStream;
import java.lang.reflect.Field;
import java.util.ArrayList;
import java.util.List;
import java.util.WeakHashMap;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Bridges the bundled Kokoro-Vietnamese backend into the existing generated Generate fragment
 * without changing the legacy Piper/Sherpa-Kokoro code path. Touch/accessibility events are
 * intercepted only while the bundled Vietnamese model is active; legacy models keep their
 * original listeners untouched.
 */
public final class VietnameseGenerateIntegration {
    public static final String MODEL_MARKER = "bundled://kokoro_vi/kokoro_vi.onnx";
    public static final String TOKENS_MARKER = "bundled://kokoro_vi/config.json";
    public static final String VOICES_MARKER = "bundled://kokoro_vi/voicepacks";
    public static final String MODEL_TYPE = "kokoro_vi";
    private static final String PREF_ACTIVE_VOICE = "active_vi_voice";

    private static final WeakHashMap<Activity, State> STATES = new WeakHashMap<>();

    private static final class State {
        View generateButton;
        View dropdown;
        EditText input;
        volatile boolean generating;
        volatile boolean cancelled;
        volatile String generatedText = "";
        volatile AudioTrack liveTrack;
        final AtomicInteger generation = new AtomicInteger();
    }

    private VietnameseGenerateIntegration() {}

    /** Make bundled Vietnamese the Generate default only when the user has no active model. */
    public static void ensureDefaultActiveModel(Context context) {
        if (context == null || !VietnameseKokoroEngine.isBundled(context)) return;
        SharedPreferences sp1 = context.getSharedPreferences("sp1", Context.MODE_PRIVATE);
        String active = sp1.getString("active_model", "");
        if (active != null && !active.trim().isEmpty()) {
            if (MODEL_MARKER.equals(active)) ensureBundledActiveMetadata(context, null);
            return;
        }

        String systemDefault = sp1.getString(
                "default_voice_Vietnamese", VietnameseKokoroVoice.DEFAULT.androidVoiceName);
        VietnameseKokoroVoice voice = VietnameseKokoroVoice.fromAndroidVoiceName(systemDefault);
        if (voice == null) voice = VietnameseKokoroVoice.DEFAULT;
        activateBundledModel(context, voice);
        TtsDiagnostics.info(context, "generate", "bundled_default_activated",
                "No previous Generate model was selected; activated Vietnamese Kokoro / " + voice.id);
    }

    /** Explicitly select bundled Vietnamese for the Generate tab. */
    public static void activateBundledModel(Context context, VietnameseKokoroVoice voice) {
        if (context == null) return;
        if (voice == null) voice = getActiveVoice(context);
        if (voice == null) voice = VietnameseKokoroVoice.DEFAULT;
        SharedPreferences sp1 = context.getSharedPreferences("sp1", Context.MODE_PRIVATE);
        sp1.edit()
                .putString("active_model", MODEL_MARKER)
                .putString("active_tokens", TOKENS_MARKER)
                .putString("active_model_type", MODEL_TYPE)
                .putString("active_model_name", "Vietnamese Kokoro")
                .putString("active_voices_bin", VOICES_MARKER)
                .putInt("active_speaker_count", VietnameseKokoroVoice.all().size())
                .putString("active_language", "Vietnamese")
                .putString(PREF_ACTIVE_VOICE, voice.id)
                .apply();
        TtsDiagnostics.info(context, "generate", "bundled_model_activated",
                "voice=" + voice.id + ", voices=" + VietnameseKokoroVoice.all().size());
    }

    private static void ensureBundledActiveMetadata(Context context, VietnameseKokoroVoice voice) {
        SharedPreferences sp1 = context.getSharedPreferences("sp1", Context.MODE_PRIVATE);
        if (voice == null) voice = getActiveVoice(context);
        if (voice == null) voice = VietnameseKokoroVoice.DEFAULT;
        sp1.edit()
                .putString("active_tokens", TOKENS_MARKER)
                .putString("active_model_type", MODEL_TYPE)
                .putString("active_model_name", "Vietnamese Kokoro")
                .putString("active_voices_bin", VOICES_MARKER)
                .putInt("active_speaker_count", VietnameseKokoroVoice.all().size())
                .putString("active_language", "Vietnamese")
                .putString(PREF_ACTIVE_VOICE, voice.id)
                .apply();
    }

    public static boolean isBundledActive(Context context) {
        if (context == null) return false;
        SharedPreferences sp1 = context.getSharedPreferences("sp1", Context.MODE_PRIVATE);
        return MODEL_MARKER.equals(sp1.getString("active_model", ""))
                || MODEL_TYPE.equalsIgnoreCase(sp1.getString("active_model_type", ""));
    }

    public static VietnameseKokoroVoice getActiveVoice(Context context) {
        if (context == null) return VietnameseKokoroVoice.DEFAULT;
        SharedPreferences sp1 = context.getSharedPreferences("sp1", Context.MODE_PRIVATE);
        VietnameseKokoroVoice voice = VietnameseKokoroVoice.fromId(
                sp1.getString(PREF_ACTIVE_VOICE, VietnameseKokoroVoice.DEFAULT.id));
        return voice == null ? VietnameseKokoroVoice.DEFAULT : voice;
    }

    public static void install(Activity activity) {
        if (activity == null || activity.getWindow() == null
                || !VietnameseKokoroEngine.isBundled(activity)) return;

        ensureDefaultActiveModel(activity);
        int generateId = id(activity, "btn_generate");
        int inputId = id(activity, "et_input");
        int voiceNameId = id(activity, "voice_name_tv");
        int dropdownId = id(activity, "opne_dropdown");
        if (generateId == 0 || inputId == 0 || voiceNameId == 0 || dropdownId == 0) return;

        View generateButton = activity.findViewById(generateId);
        EditText input = activity.findViewById(inputId);
        TextView voiceName = activity.findViewById(voiceNameId);
        View dropdown = activity.findViewById(dropdownId);
        if (generateButton == null || input == null || voiceName == null || dropdown == null) return;

        State state;
        synchronized (STATES) {
            state = STATES.get(activity);
            if (state == null) {
                state = new State();
                STATES.put(activity, state);
            }
        }

        if (state.generateButton != generateButton) {
            state.generateButton = generateButton;
            installGenerateInterceptors(activity, generateButton, state);
        }
        if (state.dropdown != dropdown) {
            state.dropdown = dropdown;
            installDropdownInterceptors(activity, dropdown, state);
        }
        if (state.input != input) {
            state.input = input;
            installInputCancellation(activity, input, state);
        }

        if (isBundledActive(activity)) {
            ensureBundledActiveMetadata(activity, getActiveVoice(activity));
            VietnameseKokoroVoice voice = getActiveVoice(activity);
            voiceName.setText("Vietnamese Kokoro • " + voice.displayName);
            voiceName.setTextColor(Color.WHITE);
            dropdown.setVisibility(View.VISIBLE);
            voiceName.setContentDescription("Vietnamese Kokoro. Giọng " + voice.displayName
                    + ". Có " + VietnameseKokoroVoice.all().size() + " giọng.");
        }
    }

    public static void onActivityDestroyed(Activity activity) {
        if (activity == null) return;
        State state;
        synchronized (STATES) { state = STATES.remove(activity); }
        if (state != null) {
            state.cancelled = true;
            state.generation.incrementAndGet();
            closeLiveTrack(state);
        }
    }

    private static void installGenerateInterceptors(Activity activity, View button, State state) {
        button.setOnTouchListener((v, event) -> {
            if (!isBundledActive(activity)) return false;
            if (event.getAction() == MotionEvent.ACTION_UP) handleGenerate(activity, state);
            return true;
        });
        button.setAccessibilityDelegate(new View.AccessibilityDelegate() {
            @Override
            public boolean performAccessibilityAction(View host, int action, Bundle args) {
                if (action == AccessibilityNodeInfo.ACTION_CLICK && isBundledActive(activity)) {
                    handleGenerate(activity, state);
                    return true;
                }
                return super.performAccessibilityAction(host, action, args);
            }
        });
        TtsDiagnostics.info(activity, "generate", "generate_bridge_installed",
                "Bundled Vietnamese Generate bridge installed without replacing legacy onClick listener.");
    }

    private static void installDropdownInterceptors(Activity activity, View dropdown, State state) {
        dropdown.setOnTouchListener((v, event) -> {
            if (!isBundledActive(activity)) return false;
            if (event.getAction() == MotionEvent.ACTION_UP) showVoicePicker(activity, state);
            return true;
        });
        dropdown.setAccessibilityDelegate(new View.AccessibilityDelegate() {
            @Override
            public boolean performAccessibilityAction(View host, int action, Bundle args) {
                if (action == AccessibilityNodeInfo.ACTION_CLICK && isBundledActive(activity)) {
                    showVoicePicker(activity, state);
                    return true;
                }
                return super.performAccessibilityAction(host, action, args);
            }
        });
    }

    private static void installInputCancellation(Activity activity, EditText input, State state) {
        input.addTextChangedListener(new TextWatcher() {
            @Override public void beforeTextChanged(CharSequence s, int start, int count, int after) {}
            @Override public void onTextChanged(CharSequence s, int start, int before, int count) {
                if (!isBundledActive(activity) || !state.generating) return;
                state.cancelled = true;
                state.generation.incrementAndGet();
                VietnameseKokoroEngine.getInstance().cancel();
                closeLiveTrack(state);
            }
            @Override public void afterTextChanged(Editable s) {}
        });
    }

    private static void showVoicePicker(Activity activity, State state) {
        TextView anchor = activity.findViewById(id(activity, "voice_name_tv"));
        if (anchor == null) return;
        List<String> labels = new ArrayList<>();
        for (VietnameseKokoroVoice voice : VietnameseKokoroVoice.all()) labels.add(voice.displayName);

        ListPopupWindow popup = new ListPopupWindow(activity);
        popup.setAnchorView(anchor);
        popup.setAdapter(new ArrayAdapter<>(activity, R.layout.dropdown_item, R.id.text1, labels));
        VietnameseKokoroVoice selected = getActiveVoice(activity);
        int selectedIndex = VietnameseKokoroVoice.all().indexOf(selected);
        if (selectedIndex >= 0) popup.setSelection(selectedIndex);
        popup.setOnItemClickListener((parent, view, position, rowId) -> {
            if (position < 0 || position >= VietnameseKokoroVoice.all().size()) return;
            VietnameseKokoroVoice voice = VietnameseKokoroVoice.all().get(position);
            activateBundledModel(activity, voice);
            VietnameseKokoroEngine.getInstance().cancel();
            state.cancelled = true;
            state.generation.incrementAndGet();
            closeLiveTrack(state);
            resetLegacyGenerateFragment(activity);
            anchor.setText("Vietnamese Kokoro • " + voice.displayName);
            TtsDiagnostics.info(activity, "generate", "voice_selected",
                    "voice=" + voice.id + ", position=" + position);
            popup.dismiss();
        });
        popup.show();
    }

    private static void handleGenerate(Activity activity, State state) {
        if (!isBundledActive(activity)) return;
        EditText input = activity.findViewById(id(activity, "et_input"));
        if (input == null) return;
        String text = input.getText() == null ? "" : input.getText().toString().trim();
        if (text.isEmpty()) {
            View root = activity.findViewById(android.R.id.content);
            if (root != null) Snackbar.make(root, "Please enter some text first.", Snackbar.LENGTH_SHORT).show();
            return;
        }

        GenerateFragmentActivity fragment = findGenerateFragment(activity);
        if (!state.generating && text.equals(state.generatedText) && fragment != null) {
            try {
                Field audioField = field(GenerateFragmentActivity.class, "audioTrack");
                Object audio = audioField.get(fragment);
                if (audio instanceof AudioTrack) {
                    fragment._toggleGeneratePlayback();
                    return;
                }
            } catch (Throwable ignored) {}
        }

        if (state.generating) {
            state.cancelled = true;
            state.generating = false;
            state.generation.incrementAndGet();
            VietnameseKokoroEngine.getInstance().cancel();
            closeLiveTrack(state);
            setStatus(activity, "CANCELED", false, true);
            TtsDiagnostics.info(activity, "generate", "cancelled_by_button", "User cancelled Vietnamese Generate.");
            return;
        }

        resetLegacyGenerateFragment(activity);
        state.cancelled = false;
        state.generating = true;
        final int myGeneration = state.generation.incrementAndGet();
        VietnameseKokoroVoice voice = getActiveVoice(activity);
        SharedPreferences sp3 = activity.getSharedPreferences("sp3", Context.MODE_PRIVATE);
        float speed = sp3.getFloat("voice_speed", 1.0f);
        float pitch = sp3.getFloat("voice_pitch", 1.0f);
        setStatus(activity, "GENERATING VOICE...", true, false);
        TtsDiagnostics.info(activity, "generate", "request",
                "voice=" + voice.id + ", chars=" + text.length() + ", speed=" + speed);

        new Thread(() -> synthesize(activity, state, myGeneration, text, voice, speed, pitch),
                "KokoroVi-Generate").start();
    }

    private static void synthesize(Activity activity, State state, int myGeneration, String text,
                                   VietnameseKokoroVoice voice, float speed, float pitch) {
        long started = System.nanoTime();
        ByteArrayOutputStream pcm = new ByteArrayOutputStream();
        try {
            int min = AudioTrack.getMinBufferSize(VietnameseKokoroEngine.SAMPLE_RATE,
                    AudioFormat.CHANNEL_OUT_MONO, AudioFormat.ENCODING_PCM_16BIT);
            if (min <= 0) min = VietnameseKokoroEngine.SAMPLE_RATE * 2;
            AudioTrack live = new AudioTrack.Builder()
                    .setAudioAttributes(new AudioAttributes.Builder()
                            .setUsage(AudioAttributes.USAGE_MEDIA)
                            .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH)
                            .build())
                    .setAudioFormat(new AudioFormat.Builder()
                            .setSampleRate(VietnameseKokoroEngine.SAMPLE_RATE)
                            .setEncoding(AudioFormat.ENCODING_PCM_16BIT)
                            .setChannelMask(AudioFormat.CHANNEL_OUT_MONO)
                            .build())
                    .setBufferSizeInBytes(min)
                    .setTransferMode(AudioTrack.MODE_STREAM)
                    .build();
            state.liveTrack = live;
            try { live.play(); } catch (Throwable ignored) {}

            boolean emitted = VietnameseKokoroEngine.getInstance().synthesizeStreaming(
                    activity.getApplicationContext(), text, voice, speed, chunk -> {
                        if (state.cancelled || state.generation.get() != myGeneration) return false;
                        try { pcm.write(chunk); } catch (Throwable ignored) {}
                        AudioTrack track = state.liveTrack;
                        if (track != null) {
                            try { track.write(chunk, 0, chunk.length, AudioTrack.WRITE_BLOCKING); }
                            catch (Throwable ignored) {}
                        }
                        return !state.cancelled && state.generation.get() == myGeneration;
                    });

            closeLiveTrack(state);
            byte[] result = pcm.toByteArray();
            boolean success = emitted && result.length > 0 && !state.cancelled
                    && state.generation.get() == myGeneration;
            long elapsedMs = (System.nanoTime() - started) / 1_000_000L;
            TtsDiagnostics.info(activity, "generate", success ? "complete" : "failed",
                    "voice=" + voice.id + ", elapsedMs=" + elapsedMs + ", pcmBytes=" + result.length
                            + ", cancelled=" + state.cancelled);

            activity.runOnUiThread(() -> {
                state.generating = false;
                if (success) {
                    state.generatedText = text;
                    publishGeneratedAudio(activity, result, text, voice, speed, pitch);
                } else if (!state.cancelled) {
                    setStatus(activity, "SYNTHESIS FAILED", false, true);
                }
            });
        } catch (Throwable t) {
            closeLiveTrack(state);
            state.generating = false;
            TtsDiagnostics.error(activity, "generate", "exception",
                    "voice=" + voice.id + ", chars=" + text.length(), t);
            activity.runOnUiThread(() -> setStatus(activity, "SYNTHESIS FAILED", false, true));
        }
    }

    private static void publishGeneratedAudio(Activity activity, byte[] pcm, String text,
                                              VietnameseKokoroVoice voice, float speed, float pitch) {
        GenerateFragmentActivity fragment = findGenerateFragment(activity);
        if (fragment == null) {
            setStatus(activity, "GENERATION COMPLETE", false, false);
            return;
        }
        try {
            AudioTrack replay = new AudioTrack.Builder()
                    .setAudioAttributes(new AudioAttributes.Builder()
                            .setUsage(AudioAttributes.USAGE_MEDIA)
                            .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH)
                            .build())
                    .setAudioFormat(new AudioFormat.Builder()
                            .setSampleRate(VietnameseKokoroEngine.SAMPLE_RATE)
                            .setEncoding(AudioFormat.ENCODING_PCM_16BIT)
                            .setChannelMask(AudioFormat.CHANNEL_OUT_MONO)
                            .build())
                    .setBufferSizeInBytes(Math.max(1, pcm.length))
                    .setTransferMode(AudioTrack.MODE_STATIC)
                    .build();
            replay.write(pcm, 0, pcm.length);

            setField(fragment, "lastGeneratedPcmData", pcm);
            setField(fragment, "lastGeneratedSampleRate", VietnameseKokoroEngine.SAMPLE_RATE);
            setField(fragment, "isAudioGeneratedForCurrentText", true);
            setField(fragment, "lastGeneratedText", text);
            setField(fragment, "isGenerating", false);
            setField(fragment, "isCancelled", false);
            setField(fragment, "audioTrack", replay);
            int voiceIndex = Math.max(0, VietnameseKokoroVoice.all().indexOf(voice));
            setField(fragment, "lastParams", new GenerationParams(
                    text, MODEL_MARKER, TOKENS_MARKER, MODEL_TYPE,
                    VOICES_MARKER + "/" + voice.id + ".f32le", voiceIndex,
                    speed, pitch, false, false));

            View idle = activity.findViewById(id(activity, "layout_idle_state"));
            View generated = activity.findViewById(id(activity, "layout_generated_state"));
            ProgressBar progress = activity.findViewById(id(activity, "progress_generating"));
            ImageView idleIcon = activity.findViewById(id(activity, "imageview65"));
            ImageView waveform = activity.findViewById(id(activity, "img_waveform"));
            View playhead = activity.findViewById(id(activity, "playhead_line"));
            TextView duration = activity.findViewById(id(activity, "tv_duration"));
            TextView status = activity.findViewById(id(activity, "textview69"));
            TextView buttonLabel = activity.findViewById(id(activity, "textview92"));
            ImageView buttonIcon = activity.findViewById(id(activity, "imageview52"));

            if (idle != null) idle.setVisibility(View.GONE);
            if (generated != null) generated.setVisibility(View.VISIBLE);
            if (progress != null) progress.setVisibility(View.GONE);
            if (idleIcon != null) idleIcon.setVisibility(View.VISIBLE);
            if (playhead != null) {
                playhead.setVisibility(View.VISIBLE);
                playhead.setTranslationX(0f);
            }
            if (status != null) {
                status.setText("GENERATION COMPLETE");
                status.setTextColor(Color.parseColor("#1D61FF"));
            }
            if (buttonLabel != null) buttonLabel.setText("Play");
            if (buttonIcon != null) buttonIcon.setImageResource(R.drawable.icon_play_circle);
            if (duration != null) {
                double seconds = (pcm.length / 2.0) / VietnameseKokoroEngine.SAMPLE_RATE;
                duration.setText(String.format(java.util.Locale.US, "0:00 / %d:%02d",
                        (int) (seconds / 60), (int) (seconds % 60)));
            }
            if (waveform != null) {
                int width = waveform.getWidth() > 0 ? waveform.getWidth() : 800;
                Bitmap bitmap = WaveformHelper.createWaveformBitmap(pcm, width, 150);
                if (bitmap != null) waveform.setImageBitmap(bitmap);
            }
            TextView voiceName = activity.findViewById(id(activity, "voice_name_tv"));
            if (voiceName != null) voiceName.setText("Vietnamese Kokoro • " + voice.displayName);
        } catch (Throwable t) {
            TtsDiagnostics.error(activity, "generate", "publish_failed", t.toString(), t);
            setStatus(activity, "GENERATION COMPLETE", false, false);
        }
    }

    private static void resetLegacyGenerateFragment(Activity activity) {
        GenerateFragmentActivity fragment = findGenerateFragment(activity);
        if (fragment != null) {
            try { fragment._forceResetToIdle(); } catch (Throwable ignored) {}
        }
    }

    private static GenerateFragmentActivity findGenerateFragment(Activity activity) {
        if (!(activity instanceof FragmentActivity)) return null;
        return findGenerateFragment(((FragmentActivity) activity).getSupportFragmentManager());
    }

    private static GenerateFragmentActivity findGenerateFragment(FragmentManager manager) {
        if (manager == null) return null;
        for (Fragment fragment : manager.getFragments()) {
            if (fragment instanceof GenerateFragmentActivity) return (GenerateFragmentActivity) fragment;
            GenerateFragmentActivity nested = findGenerateFragment(fragment.getChildFragmentManager());
            if (nested != null) return nested;
        }
        return null;
    }

    private static void setStatus(Activity activity, String text, boolean busy, boolean error) {
        ProgressBar progress = activity.findViewById(id(activity, "progress_generating"));
        ImageView idleIcon = activity.findViewById(id(activity, "imageview65"));
        TextView status = activity.findViewById(id(activity, "textview69"));
        TextView buttonLabel = activity.findViewById(id(activity, "textview92"));
        if (progress != null) progress.setVisibility(busy ? View.VISIBLE : View.GONE);
        if (idleIcon != null) idleIcon.setVisibility(busy ? View.GONE : View.VISIBLE);
        if (status != null) {
            status.setText(text);
            status.setTextColor(error ? Color.parseColor("#FF4B4B") : Color.parseColor("#1D61FF"));
        }
        if (buttonLabel != null) buttonLabel.setText(busy ? "Cancel" : "Generate");
    }

    private static void closeLiveTrack(State state) {
        AudioTrack track = state.liveTrack;
        state.liveTrack = null;
        if (track == null) return;
        try { track.pause(); } catch (Throwable ignored) {}
        try { track.flush(); } catch (Throwable ignored) {}
        try { track.stop(); } catch (Throwable ignored) {}
        try { track.release(); } catch (Throwable ignored) {}
    }

    private static Field field(Class<?> type, String name) throws Exception {
        Field field = type.getDeclaredField(name);
        field.setAccessible(true);
        return field;
    }

    private static void setField(Object target, String name, Object value) throws Exception {
        field(target.getClass(), name).set(target, value);
    }

    private static int id(Context context, String name) {
        return context.getResources().getIdentifier(name, "id", context.getPackageName());
    }
}
