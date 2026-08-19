package com.CodeBySonu.VoxSherpa.system;

import android.app.Activity;
import android.content.Intent;
import android.os.Bundle;
import android.widget.Button;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.TextView;
import android.widget.Toast;

import com.CodeBySonu.VoxSherpa.BuildConfig;
import com.CodeBySonu.VoxSherpa.vietnamese.VietnameseKokoroEngine;

/** Accessible in-app diagnostics viewer/share screen for TTS failures and performance. */
public class TtsDiagnosticsActivity extends Activity {
    private TextView output;
    private Button benchmark;
    private Button nnapi;
    private Button cpuBackend;
    private Button xnnpackBackend;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setTitle("VoxSherpa TTS Diagnostics");

        LinearLayout root = new LinearLayout(this);
        root.setOrientation(LinearLayout.VERTICAL);
        int pad = (int) (16 * getResources().getDisplayMetrics().density);
        root.setPadding(pad, pad, pad, pad);

        LinearLayout actions = new LinearLayout(this);
        actions.setOrientation(LinearLayout.HORIZONTAL);

        Button refresh = new Button(this);
        refresh.setText("Refresh");
        refresh.setOnClickListener(v -> refresh());
        actions.addView(refresh);

        Button share = new Button(this);
        share.setText("Share log");
        share.setOnClickListener(v -> shareLog());
        actions.addView(share);

        Button clear = new Button(this);
        clear.setText("Clear log");
        clear.setOnClickListener(v -> {
            TtsDiagnostics.clear(this);
            refresh();
        });
        actions.addView(clear);

        root.addView(actions);

        if (BuildConfig.KOKORO_XNNPACK_AB) {
            TextView backendTitle = new TextView(this);
            backendTitle.setText("Kokoro runtime A/B — CPU vs XNNPACK, same ORT 1.26.0");
            backendTitle.setTextSize(16f);
            backendTitle.setPadding(0, pad, 0, 0);
            root.addView(backendTitle);

            cpuBackend = new Button(this);
            cpuBackend.setContentDescription("Use CPU with ONNX Runtime 1.26.0. The session is rebuilt and warmed before testing.");
            cpuBackend.setOnClickListener(v -> switchRuntimeBackend(VietnameseKokoroEngine.BACKEND_CPU));
            root.addView(cpuBackend, new LinearLayout.LayoutParams(
                    LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT));

            xnnpackBackend = new Button(this);
            xnnpackBackend.setContentDescription("Use XNNPACK with ONNX Runtime 1.26.0. XNNPACK uses its dedicated threadpool and the same Kokoro model and voice.");
            xnnpackBackend.setOnClickListener(v -> switchRuntimeBackend(VietnameseKokoroEngine.BACKEND_XNNPACK));
            root.addView(xnnpackBackend, new LinearLayout.LayoutParams(
                    LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT));
        }

        nnapi = new Button(this);
        nnapi.setContentDescription(
                "Toggle the optional Android NNAPI accelerator for Vietnamese Kokoro. "
                        + "This is device dependent. If NNAPI cannot create a usable session, VoxSherpa safely returns to CPU mode.");
        nnapi.setOnClickListener(v -> toggleNnapi());
        root.addView(nnapi, new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.WRAP_CONTENT
        ));
        if (BuildConfig.KOKORO_XNNPACK_AB) nnapi.setVisibility(android.view.View.GONE);

        benchmark = new Button(this);
        benchmark.setText("CPU benchmark: default / 3 / 4 / 5 / 6");
        benchmark.setContentDescription(
                "Run a deep CPU benchmark for ONNX Runtime default, three, four, five, and six threads. "
                        + "Each mode uses two warmup runs and five measured runs. Speech may be temporarily blocked while the model is tested.");
        benchmark.setOnClickListener(v -> runCpuBenchmark());
        root.addView(benchmark, new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.WRAP_CONTENT
        ));
        if (BuildConfig.KOKORO_XNNPACK_AB) benchmark.setVisibility(android.view.View.GONE);

        output = new TextView(this);
        output.setTextIsSelectable(true);
        output.setTextSize(13f);
        ScrollView scroll = new ScrollView(this);
        scroll.addView(output);
        root.addView(scroll, new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                0,
                1f
        ));

        setContentView(root);
        TtsDiagnostics.info(this, "diagnostics", "opened", "Diagnostics screen opened.");
        refresh();
    }

    private void refresh() {
        String snapshot = TtsDiagnostics.snapshot(this);
        String perf = "\n\n--- Kokoro performance state ---\n"
                + VietnameseKokoroEngine.getInstance().performanceState(this);
        output.setText(snapshot + perf);
        refreshProviderButton();
        refreshRuntimeButtons();
    }

    private void refreshRuntimeButtons() {
        if (!BuildConfig.KOKORO_XNNPACK_AB || cpuBackend == null || xnnpackBackend == null) return;
        VietnameseKokoroEngine engine = VietnameseKokoroEngine.getInstance();
        String requested = engine.requestedRuntimeBackend(this);
        boolean xnnRequested = VietnameseKokoroEngine.BACKEND_XNNPACK.equals(requested);
        cpuBackend.setText(!xnnRequested ? "CPU • ORT 1.26.0 — SELECTED" : "CPU • ORT 1.26.0 — use this mode");
        xnnpackBackend.setText(xnnRequested
                ? (engine.isXnnpackActive() ? "XNNPACK • ORT 1.26.0 — SELECTED / ACTIVE"
                        : "XNNPACK • ORT 1.26.0 — SELECTED / FALLBACK CPU")
                : "XNNPACK • ORT 1.26.0 — use this mode");
    }

    private void switchRuntimeBackend(String backend) {
        if (!BuildConfig.KOKORO_XNNPACK_AB || cpuBackend == null || xnnpackBackend == null) return;
        VietnameseKokoroEngine engine = VietnameseKokoroEngine.getInstance();
        final boolean xnn = VietnameseKokoroEngine.BACKEND_XNNPACK.equals(backend);
        cpuBackend.setEnabled(false);
        xnnpackBackend.setEnabled(false);
        cpuBackend.setText(xnn ? "CPU • ORT 1.26.0" : "Switching to CPU…");
        xnnpackBackend.setText(xnn ? "Switching to XNNPACK…" : "XNNPACK • ORT 1.26.0");
        TtsDiagnostics.info(this, "provider", "ui_backend_requested",
                "User requested runtime backend=" + backend + " for controlled CPU/XNNPACK A/B.");
        new Thread(() -> {
            try {
                String active = engine.setRuntimeBackend(this, backend);
                runOnUiThread(() -> {
                    cpuBackend.setEnabled(true);
                    xnnpackBackend.setEnabled(true);
                    refresh();
                    String message = xnn && !VietnameseKokoroEngine.BACKEND_XNNPACK.equals(active)
                            ? "XNNPACK could not stay active; CPU fallback is active. See the log."
                            : "Runtime switched to " + active + " and warmed. You can test TalkBack now.";
                    Toast.makeText(this, message, Toast.LENGTH_LONG).show();
                });
            } catch (Throwable t) {
                TtsDiagnostics.error(this, "provider", "ui_backend_switch_failed", t.toString(), t);
                runOnUiThread(() -> {
                    cpuBackend.setEnabled(true);
                    xnnpackBackend.setEnabled(true);
                    refresh();
                    Toast.makeText(this, "Runtime switch failed. See TTS Logs.", Toast.LENGTH_LONG).show();
                });
            }
        }, "KokoroVi-XNNPACK-Switch").start();
    }

    private void refreshProviderButton() {
        VietnameseKokoroEngine engine = VietnameseKokoroEngine.getInstance();
        boolean requested = engine.isNnapiRequested(this);
        boolean active = engine.isNnapiActive();
        if (requested && active) nnapi.setText("NNAPI accelerator: ON (active)");
        else if (requested) nnapi.setText("NNAPI accelerator: ON (pending)");
        else nnapi.setText("NNAPI accelerator: OFF");
    }

    private void toggleNnapi() {
        VietnameseKokoroEngine engine = VietnameseKokoroEngine.getInstance();
        final boolean enable = !engine.isNnapiRequested(this);
        nnapi.setEnabled(false);
        benchmark.setEnabled(false);
        nnapi.setText(enable ? "Enabling NNAPI…" : "Disabling NNAPI…");
        TtsDiagnostics.info(this, "provider", "ui_requested",
                "User requested NNAPI=" + enable + ". Rebuilding only the ONNX session.");

        new Thread(() -> {
            try {
                boolean active = engine.setNnapiEnabled(this, enable);
                runOnUiThread(() -> {
                    nnapi.setEnabled(true);
                    benchmark.setEnabled(true);
                    refresh();
                    String message;
                    if (!enable) message = "NNAPI disabled. CPU mode is active.";
                    else if (active) message = "NNAPI enabled and active. Compare TTS latency in the log.";
                    else message = "NNAPI was not usable on this device/model. CPU mode was restored.";
                    Toast.makeText(this, message, Toast.LENGTH_LONG).show();
                });
            } catch (Throwable t) {
                TtsDiagnostics.error(this, "provider", "ui_failed", t.toString(), t);
                runOnUiThread(() -> {
                    nnapi.setEnabled(true);
                    benchmark.setEnabled(true);
                    refresh();
                    Toast.makeText(this, "NNAPI switch failed. CPU mode is kept; see TTS Logs.", Toast.LENGTH_LONG).show();
                });
            }
        }, "KokoroVi-NNAPI-Switch").start();
    }

    private void runCpuBenchmark() {
        benchmark.setEnabled(false);
        nnapi.setEnabled(false);
        benchmark.setText("CPU benchmark running…");
        TtsDiagnostics.info(this, "benchmark", "ui_requested",
                "User started lightweight ORT default/3/4/6 CPU benchmark across short/medium TalkBack workloads.");
        new Thread(() -> {
            try {
                VietnameseKokoroEngine.getInstance().benchmarkCpuThreads(this);
                runOnUiThread(() -> {
                    benchmark.setEnabled(true);
                    nnapi.setEnabled(true);
                    benchmark.setText("CPU benchmark: default / 3 / 4 / 5 / 6");
                    refresh();
                    Toast.makeText(this, "CPU benchmark complete; fastest CPU mode saved.", Toast.LENGTH_LONG).show();
                });
            } catch (Throwable t) {
                TtsDiagnostics.error(this, "benchmark", "ui_failed", t.toString(), t);
                runOnUiThread(() -> {
                    benchmark.setEnabled(true);
                    nnapi.setEnabled(true);
                    benchmark.setText("CPU benchmark: default / 3 / 4 / 5 / 6");
                    refresh();
                    Toast.makeText(this, "CPU benchmark failed. See TTS Logs.", Toast.LENGTH_LONG).show();
                });
            }
        }, "KokoroVi-CPU-Benchmark").start();
    }

    private void shareLog() {
        Intent share = new Intent(Intent.ACTION_SEND);
        share.setType("text/plain");
        share.putExtra(Intent.EXTRA_SUBJECT, "VoxSherpa TTS diagnostics");
        share.putExtra(Intent.EXTRA_TEXT,
                TtsDiagnostics.snapshot(this) + "\n\n--- Kokoro performance state ---\n"
                        + VietnameseKokoroEngine.getInstance().performanceState(this));
        startActivity(Intent.createChooser(share, "Share TTS diagnostics"));
    }
}
