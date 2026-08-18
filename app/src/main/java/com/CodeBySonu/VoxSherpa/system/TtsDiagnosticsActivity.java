package com.CodeBySonu.VoxSherpa.system;

import android.app.Activity;
import android.content.Intent;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.TextView;

/** Accessible in-app diagnostics viewer/share screen for TTS failures. */
public class TtsDiagnosticsActivity extends Activity {
    private TextView output;

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
        output.setText(TtsDiagnostics.snapshot(this));
    }

    private void shareLog() {
        Intent share = new Intent(Intent.ACTION_SEND);
        share.setType("text/plain");
        share.putExtra(Intent.EXTRA_SUBJECT, "VoxSherpa TTS diagnostics");
        share.putExtra(Intent.EXTRA_TEXT, TtsDiagnostics.snapshot(this));
        startActivity(Intent.createChooser(share, "Share TTS diagnostics"));
    }
}
