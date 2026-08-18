package com.CodeBySonu.VoxSherpa.vietnamese;

import android.content.Context;
import android.content.res.AssetFileDescriptor;

import com.CodeBySonu.VoxSherpa.system.TtsDiagnostics;

import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;

/** Installs the large model/G2P files into private storage once so native code can open them. */
public final class VietnameseKokoroAssetStore {
    private static final String ROOT = "kokoro_vi";
    private static final String MODEL = ROOT + "/kokoro_vi.onnx";
    private static final String DICTIONARY = ROOT + "/sea_g2p.bin";
    private static final String CONFIG = ROOT + "/config.json";

    public static final class Paths {
        public final File model;
        public final File dictionary;
        public final File voicepack;

        private Paths(File model, File dictionary, File voicepack) {
            this.model = model;
            this.dictionary = dictionary;
            this.voicepack = voicepack;
        }
    }

    private VietnameseKokoroAssetStore() {}

    public static boolean isBundled(Context context) {
        if (context == null) return false;
        boolean nativeAvailable = VietnameseKokoroNative.isAvailable();
        boolean assets = assetExists(context, MODEL)
                && assetExists(context, DICTIONARY)
                && assetExists(context, CONFIG)
                && assetExists(context, VietnameseKokoroVoice.DIEM_TRINH.assetPath);
        if (!nativeAvailable || !assets) {
            TtsDiagnostics.warn(context, "assets", "bundle_probe_failed",
                    "nativeAvailable=" + nativeAvailable + ", assetsPresent=" + assets
                            + ", nativeError=" + VietnameseKokoroNative.loadError());
        }
        return nativeAvailable && assets;
    }

    public static synchronized Paths ensure(Context context) throws Exception {
        if (!isBundled(context)) {
            throw new IllegalStateException("Vietnamese Kokoro Stage-1 assets/native backend are not available. nativeError="
                    + VietnameseKokoroNative.loadError());
        }
        long started = System.nanoTime();
        File root = new File(context.getFilesDir(), ROOT);
        if (!root.exists() && !root.mkdirs()) {
            throw new IllegalStateException("Cannot create Vietnamese Kokoro data directory.");
        }
        File model = copyIfNeeded(context, MODEL, new File(root, "kokoro_vi.onnx"));
        File dictionary = copyIfNeeded(context, DICTIONARY, new File(root, "sea_g2p.bin"));
        File voice = copyIfNeeded(context, VietnameseKokoroVoice.DIEM_TRINH.assetPath,
                new File(root, "diem_trinh.f32le"));
        long elapsedMs = (System.nanoTime() - started) / 1_000_000L;
        TtsDiagnostics.info(context, "assets", "ready",
                "elapsedMs=" + elapsedMs + ", modelBytes=" + model.length()
                        + ", dictionaryBytes=" + dictionary.length() + ", voiceBytes=" + voice.length());
        return new Paths(model, dictionary, voice);
    }

    private static boolean assetExists(Context context, String path) {
        try (InputStream ignored = context.getAssets().open(path)) {
            return true;
        } catch (Throwable ignored) {
            return false;
        }
    }

    private static long assetSize(Context context, String path) throws Exception {
        try (AssetFileDescriptor afd = context.getAssets().openFd(path)) {
            if (afd.getLength() > 0) return afd.getLength();
        } catch (Throwable ignored) {
        }
        try (InputStream input = context.getAssets().open(path)) {
            return input.available();
        }
    }

    private static File copyIfNeeded(Context context, String assetPath, File target) throws Exception {
        long expectedSize = assetSize(context, assetPath);
        if (expectedSize > 0 && target.exists() && target.length() == expectedSize) {
            TtsDiagnostics.info(context, "assets", "reuse",
                    assetPath + " -> " + target.getAbsolutePath() + ", bytes=" + target.length());
            return target;
        }

        long started = System.nanoTime();
        File temporary = new File(target.getParentFile(), target.getName() + ".part");
        try (InputStream input = context.getAssets().open(assetPath);
             FileOutputStream output = new FileOutputStream(temporary)) {
            byte[] buffer = new byte[1024 * 1024];
            int read;
            while ((read = input.read(buffer)) > 0) output.write(buffer, 0, read);
            output.getFD().sync();
        } catch (Throwable t) {
            TtsDiagnostics.error(context, "assets", "copy_failed", assetPath + " -> " + target, t);
            throw t;
        }
        if (target.exists() && !target.delete()) {
            throw new IllegalStateException("Cannot replace " + target.getName());
        }
        if (!temporary.renameTo(target)) {
            throw new IllegalStateException("Cannot install " + target.getName());
        }
        long elapsedMs = (System.nanoTime() - started) / 1_000_000L;
        TtsDiagnostics.info(context, "assets", "copied",
                assetPath + " -> " + target.getAbsolutePath() + ", bytes=" + target.length()
                        + ", elapsedMs=" + elapsedMs);
        return target;
    }
}
