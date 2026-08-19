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

    /**
     * Creates the ONNX session. cpuThreads=0 means ONNX Runtime default threading;
     * otherwise an explicit intra-op thread count is used.
     */
    public native boolean createEngine(String modelPath, int cpuThreads);
    public native void destroyEngine();
    public native float[] synthesize(long[] inputIds, float[] refStyle, float speed);

    /** lockWaitUs, inputPrepUs, ortRunUs, outputCopyUs, totalUs, activeThreads */
    public native long[] getLastInferenceTimingMicros();

    /**
     * Benchmarks ONNX Runtime default/4/6 intra-op CPU threads and restores a
     * live session using the winning configuration. Returns a compact JSON result.
     */
    public native String benchmarkCpuThreads(
            String modelPath,
            long[] inputIds,
            float[] refStyle,
            float speed,
            int warmupRuns,
            int measuredRuns
    );
}
