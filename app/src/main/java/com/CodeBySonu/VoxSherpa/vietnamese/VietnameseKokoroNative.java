package com.CodeBySonu.VoxSherpa.vietnamese;

/** JNI bridge for the bundled Vietnamese Kokoro backend. */
public final class VietnameseKokoroNative {
    private static final Throwable LOAD_ERROR;
    private static final Throwable QNN_GPU_LOAD_ERROR;

    static {
        Throwable qnnError = null;
        try {
            // These libraries exist only in the QNN GPU experiment APK. Loading them
            // eagerly gives Android's linker a deterministic path before ORT dlopens
            // the QNN GPU backend by name.
            System.loadLibrary("QnnSystem");
            System.loadLibrary("QnnGpu");
        } catch (Throwable t) {
            qnnError = t;
        }
        QNN_GPU_LOAD_ERROR = qnnError;

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

    public static boolean isQnnGpuLibraryAvailable() {
        return QNN_GPU_LOAD_ERROR == null;
    }

    public static String qnnGpuLoadError() {
        return QNN_GPU_LOAD_ERROR == null ? "" : QNN_GPU_LOAD_ERROR.toString();
    }

    public native long createG2p(String dictionaryPath);
    public native String phonemize(long handle, String text);
    public native void destroyG2p(long handle);

    /**
     * Creates the ONNX session. cpuThreads=0 means ONNX Runtime default threading.
     * QNN GPU has highest priority in the experiment build. If QNN session creation
     * fails, native code falls back to CPU safely. NNAPI remains available when QNN
     * is not requested.
     */
    public native boolean createEngine(
            String modelPath,
            int cpuThreads,
            boolean useNnapi,
            boolean useQnnGpu
    );
    public native void destroyEngine();
    public native float[] synthesize(long[] inputIds, float[] refStyle, float speed);

    /** Immediately terminates the currently-running Ort::Session::Run, if any. */
    public native boolean cancelActiveRun();

    /** True only when the live ONNX session actually has NNAPI enabled. */
    public native boolean isNnapiActive();

    /** True only when the live ONNX session was created with the QNN GPU backend. */
    public native boolean isQnnGpuActive();

    /** lockWaitUs, inputPrepUs, ortRunUs, outputCopyUs, totalUs, activeThreads */
    public native long[] getLastInferenceTimingMicros();

    /**
     * Benchmarks the CPU thread configurations and restores a live CPU session using
     * the winning configuration. GPU experiment builds can switch back to QNN after
     * Java receives this result.
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
