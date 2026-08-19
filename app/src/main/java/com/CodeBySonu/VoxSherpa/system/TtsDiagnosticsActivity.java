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

        benchmark = new Button(this);
        benchmark.setText("CPU benchmark: default / 4 / 6");
        benchmark.setContentDescription(
                "Run CPU benchmark for ONNX Runtime default, four threads, and six threads. "
                        + "The test may temporarily block speech while the model is reloaded.");
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
    }

    private void runCpuBenchmark() {
        benchmark.setEnabled(false);
        benchmark.setText("CPU benchmark running…");
        TtsDiagnostics.info(this, "benchmark", "ui_requested",
                "User started default/4/6 CPU thread benchmark.");
        new Thread(() -> {
            try {
                String result = VietnameseKokoroEngine.getInstance().benchmarkCpuThreads(this);
                runOnUiThread(() -> {
                    benchmark.setEnabled(true);
                    benchmark.setText("CPU benchmark: default / 4 / 6");
                    refresh();
                    Toast.makeText(this, "CPU benchmark complete; fastest mode saved.", Toast.LENGTH_LONG).show();
                });
            } catch (Throwable t) {
                TtsDiagnostics.error(this, "benchmark", "ui_failed", t.toString(), t);
                runOnUiThread(() -> {
                    benchmark.setEnabled(true);
                    benchmark.setText("CPU benchmark: default / 4 / 6");
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
