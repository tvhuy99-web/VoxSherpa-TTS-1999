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
     * Creates the ONNX session. cpuThreads=0 means ONNX Runtime default threading.
     * useNnapi requests the Android NNAPI execution provider; native code safely
     * falls back to CPU if the provider/session is unavailable for this device/model.
     */
    public native boolean createEngine(String modelPath, int cpuThreads, boolean useNnapi, boolean useXnnpack);
    public native void destroyEngine();
    public native float[] synthesize(long[] inputIds, float[] refStyle, float speed);

    /** Immediately terminates the currently-running Ort::Session::Run, if any. */
    public native boolean cancelActiveRun();

    /** True only when the live ONNX session actually has NNAPI enabled. */
    public native boolean isNnapiActive();

    /** True only when the live ONNX session was created with XNNPACK EP. */
    public native boolean isXnnpackActive();

    /** lockWaitUs, inputPrepUs, ortRunUs, outputCopyUs, totalUs, ortCpuThreads, xnnpackThreads */
    public native long[] getLastInferenceTimingMicros();

    /**
     * Benchmarks ONNX Runtime default/3/4/5/6 intra-op CPU threads and restores a
     * live CPU session using the winning configuration. Returns a compact JSON result.
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
