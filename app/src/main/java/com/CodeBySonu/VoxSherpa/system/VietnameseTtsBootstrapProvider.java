package com.CodeBySonu.VoxSherpa.system;

import android.content.ContentProvider;
import android.content.ContentValues;
import android.database.Cursor;
import android.net.Uri;

import com.CodeBySonu.VoxSherpa.vietnamese.VietnameseKokoroEngine;
import com.CodeBySonu.VoxSherpa.vietnamese.VietnameseKokoroNative;

/**
 * Starts bundled Vietnamese registration and model prewarm as soon as the app process exists.
 * This runs before Activities/Services, giving TalkBack the largest possible warmup head start.
 */
public final class VietnameseTtsBootstrapProvider extends ContentProvider {
    @Override
    public boolean onCreate() {
        android.content.Context context = getContext();
        if (context == null) return true;

        try {
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

    @Override public Cursor query(Uri uri, String[] projection, String selection,
                                  String[] selectionArgs, String sortOrder) { return null; }
    @Override public String getType(Uri uri) { return null; }
    @Override public Uri insert(Uri uri, ContentValues values) { return null; }
    @Override public int delete(Uri uri, String selection, String[] selectionArgs) { return 0; }
    @Override public int update(Uri uri, ContentValues values, String selection,
                                String[] selectionArgs) { return 0; }
}