package com.CodeBySonu.VoxSherpa;

import android.app.Activity;
import android.app.Application;
import android.content.ComponentCallbacks2;
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
import android.widget.FrameLayout;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.TextView;

import com.CodeBySonu.VoxSherpa.system.TtsDiagnostics;
import com.CodeBySonu.VoxSherpa.system.TtsDiagnosticsActivity;
import com.CodeBySonu.VoxSherpa.vietnamese.VietnameseKokoroEngine;
import com.google.android.material.card.MaterialCardView;

import java.util.Locale;
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

    @Override
    public void onTrimMemory(int level) {
        super.onTrimMemory(level);
        VietnameseKokoroEngine engine = VietnameseKokoroEngine.getInstance();
        TtsDiagnostics.info(this, "memory", "trim",
                "level=" + level + ", kokoro={" + engine.performanceState(this) + "}");

        // Android commonly reports UI_HIDDEN (20), BACKGROUND (40), and MODERATE (60)
        // simply because the app UI moved to the background. Releasing at those levels
        // defeats TalkBack latency because the 325 MB ONNX session must be loaded again.
        // Keep it hot and let Android kill the process if it truly needs the memory.
        // Explicitly release only for the strongest in-process signal or onLowMemory().
        if (level == ComponentCallbacks2.TRIM_MEMORY_RUNNING_CRITICAL
                || level == ComponentCallbacks2.TRIM_MEMORY_COMPLETE) {
            engine.releaseForMemoryPressure(this, "onTrimMemory level=" + level);
        } else {
            TtsDiagnostics.info(this, "memory", "hot_session_retained",
                    "Ignoring non-critical trim level=" + level + "; keeping Kokoro hot. "
                            + engine.performanceState(this));
        }
    }

    @Override
    public void onLowMemory() {
        super.onLowMemory();
        VietnameseKokoroEngine.getInstance().releaseForMemoryPressure(this, "onLowMemory");
    }

    private void installDiagnosticsShortcut() {
        registerActivityLifecycleCallbacks(new ActivityLifecycleCallbacks() {
            @Override
            public void onActivityCreated(Activity activity, Bundle savedInstanceState) {
                watchForSettingsUi(activity);
            }

            @Override
            public void onActivityResumed(Activity activity) {
                watchForSettingsUi(activity);
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

    private void watchForSettingsUi(Activity activity) {
        if (activity == null || activity instanceof TtsDiagnosticsActivity
                || diagnosticsWatchers.containsKey(activity) || activity.getWindow() == null) {
            return;
        }

        View root = activity.getWindow().getDecorView();
        ViewTreeObserver.OnGlobalLayoutListener listener = () -> {
            ensureSettingsIntegration(activity);
            ensureBundledModelCard(activity);
        };
        root.getViewTreeObserver().addOnGlobalLayoutListener(listener);
        diagnosticsWatchers.put(activity, listener);
        ensureSettingsIntegration(activity);
        ensureBundledModelCard(activity);
    }

    /**
     * Upgrades the Settings screen without changing its generated fragment structure:
     * - opens the real Vietnamese-aware TTS settings Activity directly (no activity-alias hop),
     * - counts the bundled Vietnamese model,
     * - keeps the direct TTS Logs shortcut beside Clear All Cache.
     */
    private void ensureSettingsIntegration(Activity activity) {
        try {
            int clearId = activity.getResources().getIdentifier(
                    "clear_civ", "id", activity.getPackageName());
            if (clearId == 0) return;

            View clearView = activity.findViewById(clearId);
            if (!(clearView instanceof MaterialCardView)) return;

            boolean bundled = VietnameseKokoroEngine.isBundled(activity);
            if (bundled) {
                int setupId = activity.getResources().getIdentifier(
                        "linear12", "id", activity.getPackageName());
                View setup = setupId == 0 ? null : activity.findViewById(setupId);
                if (setup != null && !"voxsherpa_vi_setup_direct".equals(setup.getTag())) {
                    setup.setTag("voxsherpa_vi_setup_direct");
                    setup.setOnClickListener(v -> {
                        try {
                            TtsDiagnostics.info(activity, "settings", "setup_tts_direct_open",
                                    "Opening VietnameseTtssettingsActivity directly from Setup System TTS.");
                            activity.startActivity(new Intent(activity, VietnameseTtssettingsActivity.class));
                        } catch (Throwable t) {
                            TtsDiagnostics.error(activity, "settings", "setup_tts_direct_failed",
                                    "Failed to open Vietnamese-aware TTS settings directly.", t);
                        }
                    });
                    TtsDiagnostics.info(activity, "settings", "setup_tts_route_installed",
                            "Setup System TTS now targets VietnameseTtssettingsActivity directly.");
                }

                int modelSummaryId = activity.getResources().getIdentifier(
                        "textview68", "id", activity.getPackageName());
                if (modelSummaryId != 0) {
                    TextView summary = activity.findViewById(modelSummaryId);
                    if (summary != null) {
                        String current = summary.getText() == null ? "" : summary.getText().toString();
                        if (current.startsWith("0 local models") || current.trim().isEmpty()) {
                            summary.setText("1 bundled Vietnamese model installed");
                        } else if (!current.toLowerCase(Locale.ROOT).contains("bundled")) {
                            summary.setText(current + " • 1 bundled Vietnamese model");
                        }
                    }
                }
            }

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
            TtsDiagnostics.error(activity, "settings", "settings_integration_failed",
                    "Failed to upgrade Settings for bundled Vietnamese TTS.", t);
        }
    }

    /**
     * The Models tab is backed by downloaded/imported entries in sp1.models_data, so a bundled
     * model otherwise appears as "Models Not Added". Render a small installed-only card without
     * inserting a fake bundled:// record into models_data (which would confuse legacy model code).
     */
    private void ensureBundledModelCard(Activity activity) {
        try {
            if (!VietnameseKokoroEngine.isBundled(activity)) return;

            int frameId = activity.getResources().getIdentifier(
                    "frame_layout14", "id", activity.getPackageName());
            int emptyId = activity.getResources().getIdentifier(
                    "empty_state_view", "id", activity.getPackageName());
            int recyclerId = activity.getResources().getIdentifier(
                    "recyclerview_models", "id", activity.getPackageName());
            if (frameId == 0 || emptyId == 0 || recyclerId == 0) return;

            View frameView = activity.findViewById(frameId);
            View empty = activity.findViewById(emptyId);
            View recycler = activity.findViewById(recyclerId);
            if (!(frameView instanceof FrameLayout) || empty == null || recycler == null) return;
            FrameLayout frame = (FrameLayout) frameView;

            int sortId = activity.getResources().getIdentifier("sort_tv", "id", activity.getPackageName());
            TextView sort = sortId == 0 ? null : activity.findViewById(sortId);
            String filter = sort == null || sort.getText() == null
                    ? "all models" : sort.getText().toString().trim().toLowerCase(Locale.ROOT);
            boolean filterAllowsBundled = filter.equals("all models")
                    || filter.equals("installed")
                    || filter.contains("vietnamese")
                    || filter.contains("tiếng việt");
            boolean legacyListEmpty = recycler.getVisibility() != View.VISIBLE;
            boolean shouldShow = legacyListEmpty && filterAllowsBundled;

            View existing = frame.findViewWithTag("voxsherpa_bundled_vi_model_card");
            if (!shouldShow) {
                if (existing != null) existing.setVisibility(View.GONE);
                if (legacyListEmpty) empty.setVisibility(View.VISIBLE);
                return;
            }

            empty.setVisibility(View.GONE);
            int countId = activity.getResources().getIdentifier(
                    "model_count_tv", "id", activity.getPackageName());
            if (countId != 0) {
                TextView count = activity.findViewById(countId);
                if (count != null) count.setText("MODELS LIST (1)");
            }

            if (existing != null) {
                existing.setVisibility(View.VISIBLE);
                return;
            }

            MaterialCardView card = new MaterialCardView(activity);
            card.setTag("voxsherpa_bundled_vi_model_card");
            card.setCardBackgroundColor(Color.parseColor("#131B2D"));
            card.setStrokeColor(Color.parseColor("#1D61FF"));
            card.setStrokeWidth(dp(activity, 1));
            card.setRadius(dp(activity, 14));
            card.setCardElevation(0f);
            card.setClickable(false);
            card.setFocusable(true);
            card.setContentDescription(
                    "Vietnamese Kokoro. Diễm Trinh. Bundled, offline, installed.");

            LinearLayout body = new LinearLayout(activity);
            body.setOrientation(LinearLayout.VERTICAL);
            body.setPadding(dp(activity, 16), dp(activity, 14), dp(activity, 16), dp(activity, 14));
            body.setImportantForAccessibility(View.IMPORTANT_FOR_ACCESSIBILITY_NO);

            TextView title = new TextView(activity);
            title.setText("Vietnamese Kokoro");
            title.setTextSize(16f);
            title.setTypeface(Typeface.DEFAULT_BOLD);
            title.setTextColor(Color.WHITE);
            body.addView(title);

            TextView subtitle = new TextView(activity);
            subtitle.setText("Diễm Trinh • Bundled • Offline • Installed");
            subtitle.setTextSize(13f);
            subtitle.setTextColor(Color.parseColor("#A0AEC0"));
            LinearLayout.LayoutParams subtitleParams = new LinearLayout.LayoutParams(
                    ViewGroup.LayoutParams.WRAP_CONTENT, ViewGroup.LayoutParams.WRAP_CONTENT);
            subtitleParams.topMargin = dp(activity, 6);
            body.addView(subtitle, subtitleParams);

            card.addView(body, new MaterialCardView.LayoutParams(
                    ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT));
            FrameLayout.LayoutParams cardParams = new FrameLayout.LayoutParams(
                    ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT,
                    Gravity.TOP);
            cardParams.setMargins(0, dp(activity, 4), 0, 0);
            frame.addView(card, cardParams);
            TtsDiagnostics.info(activity, "models", "bundled_model_card_added",
                    "Models tab now exposes bundled Vietnamese Kokoro / Diem Trinh as installed.");
        } catch (Throwable t) {
            TtsDiagnostics.error(activity, "models", "bundled_model_card_failed",
                    "Failed to expose bundled Vietnamese model in Models tab.", t);
        }
    }

    private static int dp(Context context, int value) {
        return Math.round(value * context.getResources().getDisplayMetrics().density);
    }
}
