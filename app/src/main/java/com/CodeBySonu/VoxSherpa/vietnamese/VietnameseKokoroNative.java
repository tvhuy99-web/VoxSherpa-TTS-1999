package com.CodeBySonu.VoxSherpa.vietnamese;

/** JNI bridge for the bundled Vietnamese Kokoro backend. */
public final class VietnameseKokoroNative {
    private static final Throwable LOAD_ERROR;

    static {
        Throwable error = null;
        try {
            System.loadLibrary("onnxruntime");
            System.loadLibrary("sea_g2p_android");
            System.loadLibrary("voxsherpa_kokoro_vi");
        } catch (Throwable t) {
            error = t;
        }
        LOAD_ERROR = error;
    }

    public static boolean isAvailable() {
        return LOAD_ERROR == null;
    }

    public static String loadError() {
        return LOAD_ERROR == null ? "" : LOAD_ERROR.toString();
    }

    public native long createG2p(String dictionaryPath);
    public native String phonemize(long handle, String text);
    public native void destroyG2p(long handle);
    public native boolean createEngine(String modelPath);
    public native void destroyEngine();
    public native float[] synthesize(long[] inputIds, float[] refStyle, float speed);
}
