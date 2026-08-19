package com.CodeBySonu.VoxSherpa.system;

import android.app.Activity;
import android.content.Context;
import android.content.SharedPreferences;
import android.os.Bundle;
import android.view.View;
import android.view.accessibility.AccessibilityNodeInfo;
import android.widget.SeekBar;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.fragment.app.Fragment;
import androidx.fragment.app.FragmentActivity;
import androidx.fragment.app.FragmentManager;

import com.CodeBySonu.VoxSherpa.R;
import com.CodeBySonu.VoxSherpa.SettingFragmentActivity;

import java.util.WeakHashMap;

/**
 * Makes the generated Settings sliders persist when they are changed through accessibility.
 *
 * The original generated fragment saves Speed/Pitch/Silence only from onStopTrackingTouch().
 * TalkBack changes SeekBar progress through accessibility actions, so that touch callback may
 * never run even though the visible value changes. This bridge leaves the fragment's existing
 * SeekBar listeners untouched and only persists after an accessibility action changes progress.
 */
public final class SettingsAccessibilityPersistence {
    private static final String PREFS = "sp3";

    private static final WeakHashMap<FragmentActivity, FragmentManager.FragmentLifecycleCallbacks>
            CALLBACKS = new WeakHashMap<>();
    private static final WeakHashMap<SeekBar, Boolean> BOUND = new WeakHashMap<>();

    private SettingsAccessibilityPersistence() {}

    public static void install(Activity activity) {
        if (!(activity instanceof FragmentActivity)) return;
        FragmentActivity host = (FragmentActivity) activity;

        synchronized (CALLBACKS) {
            if (CALLBACKS.containsKey(host)) return;

            FragmentManager.FragmentLifecycleCallbacks callbacks =
                    new FragmentManager.FragmentLifecycleCallbacks() {
                        @Override
                        public void onFragmentViewCreated(@NonNull FragmentManager fm,
                                                          @NonNull Fragment fragment,
                                                          @NonNull View view,
                                                          @Nullable Bundle savedInstanceState) {
                            if (fragment instanceof SettingFragmentActivity) {
                                bindSettingsView(host, view);
                            }
                        }
                    };
            host.getSupportFragmentManager().registerFragmentLifecycleCallbacks(callbacks, true);
            CALLBACKS.put(host, callbacks);
        }

        // Covers a Settings fragment that was restored before this lifecycle hook was installed.
        host.getWindow().getDecorView().post(() -> {
            try {
                for (Fragment fragment : host.getSupportFragmentManager().getFragments()) {
                    if (fragment instanceof SettingFragmentActivity && fragment.getView() != null) {
                        bindSettingsView(host, fragment.getView());
                    }
                }
            } catch (Throwable t) {
                TtsDiagnostics.error(host, "settings", "accessibility_slider_probe_failed",
                        t.toString(), t);
            }
        });
    }

    public static void uninstall(Activity activity) {
        if (!(activity instanceof FragmentActivity)) return;
        FragmentActivity host = (FragmentActivity) activity;
        FragmentManager.FragmentLifecycleCallbacks callbacks;
        synchronized (CALLBACKS) {
            callbacks = CALLBACKS.remove(host);
        }
        if (callbacks != null) {
            try {
                host.getSupportFragmentManager().unregisterFragmentLifecycleCallbacks(callbacks);
            } catch (Throwable ignored) {}
        }
    }

    private static void bindSettingsView(Context context, View root) {
        bindSeekBar(context, root.findViewById(R.id.seekbar_pitch), "voice_pitch", SliderKind.PITCH);
        bindSeekBar(context, root.findViewById(R.id.seekbar_speed), "voice_speed", SliderKind.SPEED);
        bindSeekBar(context, root.findViewById(R.id.seekbar_silence_pause), "silence_scale", SliderKind.SILENCE);
    }

    private static void bindSeekBar(Context context, SeekBar seekBar, String key, SliderKind kind) {
        if (seekBar == null) return;
        synchronized (BOUND) {
            if (BOUND.containsKey(seekBar)) return;
            BOUND.put(seekBar, Boolean.TRUE);
        }

        final View.AccessibilityDelegate previous = seekBar.getAccessibilityDelegate();
        seekBar.setAccessibilityDelegate(new View.AccessibilityDelegate() {
            @Override
            public boolean performAccessibilityAction(View host, int action, Bundle args) {
                int before = seekBar.getProgress();
                boolean handled;
                if (previous != null) {
                    handled = previous.performAccessibilityAction(host, action, args);
                } else {
                    handled = super.performAccessibilityAction(host, action, args);
                }

                int after = seekBar.getProgress();
                if (handled && after != before && isProgressAction(action)) {
                    persist(context, key, kind, after, action);
                }
                return handled;
            }
        });

        TtsDiagnostics.info(context, "settings", "accessibility_slider_bound",
                "key=" + key + ", progress=" + seekBar.getProgress());
    }

    private static boolean isProgressAction(int action) {
        return action == AccessibilityNodeInfo.ACTION_SET_PROGRESS
                || action == AccessibilityNodeInfo.ACTION_SCROLL_FORWARD
                || action == AccessibilityNodeInfo.ACTION_SCROLL_BACKWARD;
    }

    private static void persist(Context context, String key, SliderKind kind,
                                int progress, int action) {
        float value;
        if (kind == SliderKind.SILENCE) {
            value = clamp(progress / 100.0f, 0.0f, 1.0f);
        } else if (progress <= 50) {
            value = 0.25f + (progress / 50.0f) * 0.75f;
        } else {
            value = 1.0f + ((progress - 50) / 50.0f);
        }
        value = kind == SliderKind.SILENCE
                ? clamp(value, 0.0f, 1.0f)
                : clamp(value, 0.25f, 2.0f);

        SharedPreferences prefs = context.getSharedPreferences(PREFS, Context.MODE_PRIVATE);
        prefs.edit().putFloat(key, value).apply();
        TtsDiagnostics.info(context, "settings", "accessibility_slider_saved",
                "key=" + key + ", progress=" + progress + ", value=" + value
                        + ", action=" + action);
    }

    private static float clamp(float value, float min, float max) {
        return Math.max(min, Math.min(max, value));
    }

    private enum SliderKind { PITCH, SPEED, SILENCE }
}
