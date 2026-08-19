package com.CodeBySonu.VoxSherpa.system;

import android.app.Activity;
import android.app.Application;
import android.content.ContentProvider;
import android.content.ContentValues;
import android.content.Intent;
import android.database.Cursor;
import android.net.Uri;
import android.os.Bundle;

import com.CodeBySonu.VoxSherpa.TtssettingsActivity;
import com.CodeBySonu.VoxSherpa.VietnameseTtssettingsActivity;
import com.CodeBySonu.VoxSherpa.vietnamese.VietnameseKokoroEngine;
import com.CodeBySonu.VoxSherpa.vietnamese.VietnameseKokoroNative;

/**
 * Starts bundled Vietnamese registration and model prewarm as soon as the app process exists.
 * It also guarantees that every legacy TTS-settings launch converges on the unified
 * Vietnamese-aware screen, even if generated Fragment code later restores an old listener.
 */
public final class VietnameseTtsBootstrapProvider extends ContentProvider {
    private static volatile boolean routeGuardInstalled = false;

    @Override
    public boolean onCreate() {
        android.content.Context context = getContext();
        if (context == null) return true;

        try {
            installTtsSettingsRouteGuard(context);
            TtsDefaultHelper.syncDefaultVoices(context);
            boolean bundled = VietnameseKokoroEngine.isBundled(context);
            boolean nativeAvailable = VietnameseKokoroNative.isAvailable();
            TtsDiagnostics.info(context, "bootstrap", "process_started",
                    "bundled=" + bundled + ", nativeAvailable=" + nativeAvailable
                            + ", loadError=" + VietnameseKokoroNative.loadError());
            if (bundled && nativeAvailable) {
                VietnameseKokoroEngine.getInstance().prewarmAsync(context, "process_bootstrap");
            }
        } catch (Throwable t) {
            TtsDiagnostics.error(context, "bootstrap", "failed", t.toString(), t);
        }
        return true;
    }

    private static synchronized void installTtsSettingsRouteGuard(android.content.Context context) {
        if (routeGuardInstalled) return;
        android.content.Context appContext = context.getApplicationContext();
        if (!(appContext instanceof Application)) {
            TtsDiagnostics.warn(context, "settings", "route_guard_unavailable",
                    "Application context unavailable; cannot install TTS settings route guard.");
            return;
        }

        Application app = (Application) appContext;
        app.registerActivityLifecycleCallbacks(new Application.ActivityLifecycleCallbacks() {
            @Override
            public void onActivityCreated(Activity activity, Bundle savedInstanceState) {
                SettingsAccessibilityPersistence.install(activity);

                // Exact-class check is intentional: VietnameseTtssettingsActivity extends
                // TtssettingsActivity and must be allowed to render normally.
                if (activity == null || activity.getClass() != TtssettingsActivity.class) return;

                TtsDiagnostics.info(activity, "settings", "legacy_tts_settings_redirect",
                        "Legacy TtssettingsActivity was launched; redirecting to the unified Vietnamese-aware screen.");

                activity.getWindow().getDecorView().post(() -> {
                    if (activity.isFinishing() || activity.isDestroyed()) return;
                    try {
                        Intent intent = new Intent(activity, VietnameseTtssettingsActivity.class);
                        if (activity.getIntent() != null && activity.getIntent().getExtras() != null) {
                            intent.putExtras(activity.getIntent().getExtras());
                        }
                        activity.startActivity(intent);
                        activity.finish();
                        activity.overridePendingTransition(0, 0);
                    } catch (Throwable t) {
                        TtsDiagnostics.error(activity, "settings", "legacy_tts_settings_redirect_failed",
                                "Failed to redirect legacy TTS settings to unified Vietnamese screen.", t);
                    }
                });
            }

            @Override public void onActivityStarted(Activity activity) {}
            @Override public void onActivityResumed(Activity activity) {}
            @Override public void onActivityPaused(Activity activity) {}
            @Override public void onActivityStopped(Activity activity) {}
            @Override public void onActivitySaveInstanceState(Activity activity, Bundle outState) {}
            @Override public void onActivityDestroyed(Activity activity) {
                SettingsAccessibilityPersistence.uninstall(activity);
            }
        });

        routeGuardInstalled = true;
        TtsDiagnostics.info(context, "settings", "route_guard_installed",
                "All legacy TtssettingsActivity launches will converge on VietnameseTtssettingsActivity.");
    }

    @Override public Cursor query(Uri uri, String[] projection, String selection,
                                  String[] selectionArgs, String sortOrder) { return null; }
    @Override public String getType(Uri uri) { return null; }
    @Override public Uri insert(Uri uri, ContentValues values) { return null; }
    @Override public int delete(Uri uri, String selection, String[] selectionArgs) { return 0; }
    @Override public int update(Uri uri, ContentValues values, String selection,
                                String[] selectionArgs) { return 0; }
}
