package com.CodeBySonu.VoxSherpa.system;

import android.app.Activity;
import android.content.Intent;
import android.os.Bundle;
import android.widget.Button;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.TextView;
import android.widget.Toast;

import com.CodeBySonu.VoxSherpa.vietnamese.VietnameseKokoroEngine;

/** Accessible in-app diagnostics viewer/share screen for TTS failures and performance. */
public class TtsDiagnosticsActivity extends Activity {
    private TextView output;
    private Button benchmark;
    private Button nnapi;

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

        nnapi = new Button(this);
        nnapi.setContentDescription(
                "Toggle the optional Android NNAPI accelerator for Vietnamese Kokoro. "
                        + "This is device dependent. If NNAPI cannot create a usable session, VoxSherpa safely returns to CPU mode.");
        nnapi.setOnClickListener(v -> toggleNnapi());
        root.addView(nnapi, new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.WRAP_CONTENT
        ));

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
