package com.CodeBySonu.VoxSherpa.system;

import android.content.Context;
import android.content.pm.PackageInfo;
import android.content.res.AssetFileDescriptor;
import android.os.Build;
import android.util.Log;

import com.CodeBySonu.VoxSherpa.VietnameseGenerateIntegration;
import com.CodeBySonu.VoxSherpa.vietnamese.VietnameseKokoroNative;
import com.CodeBySonu.VoxSherpa.vietnamese.VietnameseKokoroVoice;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.nio.charset.StandardCharsets;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Locale;

/** Persistent diagnostics for the complete Android System-TTS boundary. */
public final class TtsDiagnostics {
    private static final String TAG = "VoxSherpaTtsDiag";
    private static final Object LOCK = new Object();
    private static final long MAX_LOG_BYTES = 2L * 1024L * 1024L;
    private static final String DIR = "diagnostics";
    private static final String FILE = "tts-diagnostics.log";

    private TtsDiagnostics() {}

    public static void info(Context context, String area, String event, String message) {
        write(context, "INFO", area, event, message, null);
    }

    public static void warn(Context context, String area, String event, String message) {
        write(context, "WARN", area, event, message, null);
    }

    public static void error(Context context, String area, String event, String message, Throwable error) {
        write(context, "ERROR", area, event, message, error);
    }

    private static void write(Context context, String level, String area, String event,
                              String message, Throwable error) {
        String safeMessage = message == null ? "" : message.replace('\n', ' ');
        String line = timestamp() + " | " + level + " | " + area + " | " + event + " | " + safeMessage;
        if (error != null) line += " | " + Log.getStackTraceString(error).replace('\n', ' ');
        if ("ERROR".equals(level)) Log.e(TAG, line, error);
        else if ("WARN".equals(level)) Log.w(TAG, line);
        else Log.i(TAG, line);
        if (context == null) return;
        synchronized (LOCK) {
            try {
                File file = getLogFile(context);
                if (file.exists() && file.length() >= MAX_LOG_BYTES) {
                    File old = new File(file.getParentFile(), FILE + ".1");
                    if (old.exists()) old.delete();
                    file.renameTo(old);
                }
                try (FileOutputStream out = new FileOutputStream(file, true)) {
                    out.write((line + "\n").getBytes(StandardCharsets.UTF_8));
                }
            } catch (Throwable t) {
                Log.e(TAG, "Unable to persist diagnostics", t);
            }
        }
    }

    public static File getLogFile(Context context) {
        File dir = new File(context.getFilesDir(), DIR);
        if (!dir.exists()) dir.mkdirs();
        return new File(dir, FILE);
    }

    public static void clear(Context context) {
        synchronized (LOCK) {
            File file = getLogFile(context);
            if (file.exists()) file.delete();
            File old = new File(file.getParentFile(), FILE + ".1");
            if (old.exists()) old.delete();
        }
        info(context, "diagnostics", "cleared", "Diagnostic log was cleared by the user.");
    }

    public static String read(Context context) {
        synchronized (LOCK) {
            try {
                File file = getLogFile(context);
                if (!file.exists()) return "No persisted TTS events yet.";
                try (FileInputStream input = new FileInputStream(file)) {
                    ByteArrayOutputStream out = new ByteArrayOutputStream();
                    byte[] buffer = new byte[8192];
                    int read;
                    while ((read = input.read(buffer)) > 0) out.write(buffer, 0, read);
                    return out.toString("UTF-8");
                }
            } catch (Throwable t) {
                return "Unable to read diagnostics: " + t;
            }
        }
    }

    public static String snapshot(Context context) {
        StringBuilder out = new StringBuilder();
        out.append("VoxSherpa TTS diagnostics\n");
        out.append("Generated: ").append(timestamp()).append('\n');
        out.append("Device: ").append(Build.MANUFACTURER).append(' ').append(Build.MODEL).append('\n');
        out.append("Android: ").append(Build.VERSION.RELEASE).append(" (SDK ").append(Build.VERSION.SDK_INT).append(")\n");
        out.append("ABIs: ").append(java.util.Arrays.toString(Build.SUPPORTED_ABIS)).append('\n');
        try {
            PackageInfo info = context.getPackageManager().getPackageInfo(context.getPackageName(), 0);
            out.append("App: ").append(info.versionName).append(" code=").append(info.getLongVersionCode()).append('\n');
        } catch (Throwable ignored) {}
        out.append("Native Vietnamese backend: ")
                .append(VietnameseKokoroNative.isAvailable() ? "AVAILABLE" : "FAILED").append('\n');
        if (!VietnameseKokoroNative.isAvailable()) {
            out.append("Native load error: ").append(VietnameseKokoroNative.loadError()).append('\n');
        }
        appendAsset(out, context, "kokoro_vi/kokoro_vi.onnx");
        appendAsset(out, context, "kokoro_vi/sea_g2p.bin");
        appendAsset(out, context, "kokoro_vi/config.json");

        int voiceOk = 0;
        for (VietnameseKokoroVoice voice : VietnameseKokoroVoice.all()) {
            if (assetExists(context, voice.assetPath)) voiceOk++;
        }
        out.append("Bundled Vietnamese voices: ").append(voiceOk).append('/')
                .append(VietnameseKokoroVoice.all().size()).append(" OK\n");
        for (VietnameseKokoroVoice voice : VietnameseKokoroVoice.all()) {
            out.append("Voice ").append(voice.id).append(" (" + voice.displayName + "): ")
                    .append(assetExists(context, voice.assetPath) ? "OK" : "MISSING").append('\n');
        }
        String systemVoice = context.getSharedPreferences("sp1", Context.MODE_PRIVATE)
                .getString("default_voice_Vietnamese", VietnameseKokoroVoice.DEFAULT.androidVoiceName);
        out.append("System Vietnamese default: ").append(systemVoice).append('\n');
        out.append("Generate Vietnamese active: ").append(VietnameseGenerateIntegration.isBundledActive(context));
        if (VietnameseGenerateIntegration.isBundledActive(context)) {
            out.append(" voice=").append(VietnameseGenerateIntegration.getActiveVoice(context).id);
        }
        out.append('\n');

        File log = getLogFile(context);
        out.append("Persistent log: ").append(log.getAbsolutePath())
                .append(" bytes=").append(log.exists() ? log.length() : 0).append("\n\n");
        out.append("--- Recent events ---\n");
        out.append(read(context));
        return out.toString();
    }

    private static boolean assetExists(Context context, String path) {
        try (AssetFileDescriptor afd = context.getAssets().openFd(path)) {
            return afd.getLength() >= 0;
        } catch (Throwable ignored) {
            try {
                context.getAssets().open(path).close();
                return true;
            } catch (Throwable missing) {
                return false;
            }
        }
    }

    private static void appendAsset(StringBuilder out, Context context, String path) {
        try (AssetFileDescriptor afd = context.getAssets().openFd(path)) {
            out.append("Asset ").append(path).append(": OK, ").append(afd.getLength()).append(" bytes\n");
        } catch (Throwable t) {
            try {
                context.getAssets().open(path).close();
                out.append("Asset ").append(path).append(": OK, size unavailable\n");
            } catch (Throwable missing) {
                out.append("Asset ").append(path).append(": MISSING/ERROR, ").append(missing).append('\n');
            }
        }
    }

    private static String timestamp() {
        return new SimpleDateFormat("yyyy-MM-dd HH:mm:ss.SSS Z", Locale.US).format(new Date());
    }
}
