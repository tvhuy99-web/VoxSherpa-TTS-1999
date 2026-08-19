package com.CodeBySonu.VoxSherpa;

import android.app.Activity;
import android.app.Application;
import android.content.Context;
import android.content.Intent;
import android.content.res.ColorStateList;
import android.graphics.Color;
import android.graphics.Typeface;
import android.os.Bundle;
import android.os.Process;
import android.util.Log;
import android.view.Gravity;
import android.view.View;
import android.view.ViewGroup;
import android.view.ViewParent;
import android.view.ViewTreeObserver;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.TextView;

import com.CodeBySonu.VoxSherpa.system.TtsDiagnostics;
import com.CodeBySonu.VoxSherpa.system.TtsDiagnosticsActivity;
import com.google.android.material.card.MaterialCardView;

import java.util.WeakHashMap;

public class SketchApplication extends Application {

    private static Context mApplicationContext;
    private final WeakHashMap<Activity, ViewTreeObserver.OnGlobalLayoutListener> diagnosticsWatchers =
            new WeakHashMap<>();

    public static Context getContext() {
        return mApplicationContext;
    }

    @Override
    public void onCreate() {
        mApplicationContext = getApplicationContext();

        Thread.setDefaultUncaughtExceptionHandler(
                new Thread.UncaughtExceptionHandler() {
                    @Override
                    public void uncaughtException(Thread thread, Throwable throwable) {
                        Intent intent = new Intent(getApplicationContext(), DebugActivity.class);
                        intent.setFlags(Intent.FLAG_ACTIVITY_NEW_TASK | Intent.FLAG_ACTIVITY_CLEAR_TASK);
                        intent.putExtra("error", Log.getStackTraceString(throwable));
                        startActivity(intent);
                        Process.killProcess(Process.myPid());
                        System.exit(1);
                    }
                });
        super.onCreate();
        installDiagnosticsShortcut();
    }

    /**
     * Keeps a directly accessible TTS Logs button beside the existing Clear All Cache button.
     * The Settings screen is a Fragment inside MainActivity, so the watcher waits until that
     * Fragment view exists and then performs the one-time layout upgrade for that view instance.
     */
    private void installDiagnosticsShortcut() {
        registerActivityLifecycleCallbacks(new ActivityLifecycleCallbacks() {
            @Override
            public void onActivityCreated(Activity activity, Bundle savedInstanceState) {
                watchForSettingsStorageRow(activity);
            }

            @Override
            public void onActivityResumed(Activity activity) {
                watchForSettingsStorageRow(activity);
            }

            @Override
            public void onActivityDestroyed(Activity activity) {
                ViewTreeObserver.OnGlobalLayoutListener listener = diagnosticsWatchers.remove(activity);
                if (listener == null || activity == null || activity.getWindow() == null) return;
                View root = activity.getWindow().getDecorView();
                ViewTreeObserver observer = root.getViewTreeObserver();
                if (observer.isAlive()) observer.removeOnGlobalLayoutListener(listener);
            }

            @Override public void onActivityStarted(Activity activity) {}
            @Override public void onActivityPaused(Activity activity) {}
            @Override public void onActivityStopped(Activity activity) {}
            @Override public void onActivitySaveInstanceState(Activity activity, Bundle outState) {}
        });
    }

    private void watchForSettingsStorageRow(Activity activity) {
        if (activity == null || activity instanceof TtsDiagnosticsActivity
                || diagnosticsWatchers.containsKey(activity) || activity.getWindow() == null) {
            return;
        }

        View root = activity.getWindow().getDecorView();
        ViewTreeObserver.OnGlobalLayoutListener listener = () -> ensureDiagnosticsButton(activity);
        root.getViewTreeObserver().addOnGlobalLayoutListener(listener);
        diagnosticsWatchers.put(activity, listener);
        ensureDiagnosticsButton(activity);
    }

    private void ensureDiagnosticsButton(Activity activity) {
        try {
            int clearId = activity.getResources().getIdentifier(
                    "clear_civ", "id", activity.getPackageName());
            if (clearId == 0) return;

            View clearView = activity.findViewById(clearId);
            if (!(clearView instanceof MaterialCardView)) return;

            ViewParent currentParent = clearView.getParent();
            if (!(currentParent instanceof LinearLayout)) return;
            if ("voxsherpa_tts_logs_row".equals(((LinearLayout) currentParent).getTag())) return;

            LinearLayout originalParent = (LinearLayout) currentParent;
            int originalIndex = originalParent.indexOfChild(clearView);
            if (originalIndex < 0) return;

            ViewGroup.LayoutParams oldParams = clearView.getLayoutParams();
            int oldTopMargin = 0;
            int oldBottomMargin = 0;
            int oldLeftMargin = 0;
            int oldRightMargin = 0;
            if (oldParams instanceof ViewGroup.MarginLayoutParams) {
                ViewGroup.MarginLayoutParams margins = (ViewGroup.MarginLayoutParams) oldParams;
                oldTopMargin = margins.topMargin;
                oldBottomMargin = margins.bottomMargin;
                oldLeftMargin = margins.leftMargin;
                oldRightMargin = margins.rightMargin;
            }

            originalParent.removeView(clearView);

            LinearLayout row = new LinearLayout(activity);
            row.setTag("voxsherpa_tts_logs_row");
            row.setOrientation(LinearLayout.HORIZONTAL);
            row.setGravity(Gravity.CENTER_VERTICAL);

            LinearLayout.LayoutParams rowParams = new LinearLayout.LayoutParams(
                    ViewGroup.LayoutParams.MATCH_PARENT, dp(activity, 50));
            rowParams.setMargins(oldLeftMargin, oldTopMargin, oldRightMargin, oldBottomMargin);
            originalParent.addView(row, originalIndex, rowParams);

            LinearLayout.LayoutParams clearParams = new LinearLayout.LayoutParams(
                    0, ViewGroup.LayoutParams.MATCH_PARENT, 1f);
            row.addView(clearView, clearParams);

            final int accent = Color.parseColor("#1D61FF");
            MaterialCardView logsCard = new MaterialCardView(activity);
            logsCard.setCardBackgroundColor(Color.TRANSPARENT);
            logsCard.setRadius(dp(activity, 12));
            logsCard.setStrokeColor(accent);
            logsCard.setStrokeWidth(dp(activity, 1));
            logsCard.setCardElevation(0f);
            logsCard.setClickable(true);
            logsCard.setFocusable(true);
            logsCard.setContentDescription("TTS Logs. Open detailed TTS diagnostics and share logs.");

            LinearLayout.LayoutParams logsParams = new LinearLayout.LayoutParams(
                    0, ViewGroup.LayoutParams.MATCH_PARENT, 1f);
            logsParams.setMarginStart(dp(activity, 10));

            LinearLayout content = new LinearLayout(activity);
            content.setOrientation(LinearLayout.HORIZONTAL);
            content.setGravity(Gravity.CENTER);
            content.setImportantForAccessibility(View.IMPORTANT_FOR_ACCESSIBILITY_NO);

            ImageView icon = new ImageView(activity);
            icon.setImageResource(R.drawable.ic_info);
            icon.setImageTintList(ColorStateList.valueOf(accent));
            icon.setImportantForAccessibility(View.IMPORTANT_FOR_ACCESSIBILITY_NO);
            content.addView(icon, new LinearLayout.LayoutParams(dp(activity, 20), dp(activity, 20)));

            TextView label = new TextView(activity);
            label.setText("TTS Logs");
            label.setTextSize(14f);
            label.setTypeface(Typeface.DEFAULT_BOLD);
            label.setTextColor(accent);
            label.setImportantForAccessibility(View.IMPORTANT_FOR_ACCESSIBILITY_NO);
            LinearLayout.LayoutParams labelParams = new LinearLayout.LayoutParams(
                    ViewGroup.LayoutParams.WRAP_CONTENT, ViewGroup.LayoutParams.WRAP_CONTENT);
            labelParams.setMarginStart(dp(activity, 10));
            content.addView(label, labelParams);

            logsCard.addView(content, new MaterialCardView.LayoutParams(
                    ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.MATCH_PARENT));
            logsCard.setOnClickListener(v -> {
                TtsDiagnostics.info(activity, "settings", "direct_logs_button_clicked",
                        "TTS Logs opened from the Storage section beside Clear All Cache.");
                activity.startActivity(new Intent(activity, TtsDiagnosticsActivity.class));
            });
            row.addView(logsCard, logsParams);

            TtsDiagnostics.info(activity, "settings", "direct_logs_button_installed",
                    "TTS Logs button installed beside Clear All Cache.");
        } catch (Throwable t) {
            TtsDiagnostics.error(activity, "settings", "direct_logs_button_failed",
                    "Failed to install TTS Logs button beside Clear All Cache.", t);
        }
    }

    private static int dp(Context context, int value) {
        return Math.round(value * context.getResources().getDisplayMetrics().density);
    }
}
