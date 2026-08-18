package com.CodeBySonu.VoxSherpa;

import android.content.Intent;
import android.os.Bundle;
import android.view.View;
import android.widget.PopupMenu;
import android.widget.TextView;

import com.CodeBySonu.VoxSherpa.system.TtsDiagnostics;
import com.CodeBySonu.VoxSherpa.system.TtsDiagnosticsActivity;
import com.CodeBySonu.VoxSherpa.vietnamese.VietnameseKokoroEngine;
import com.CodeBySonu.VoxSherpa.vietnamese.VietnameseKokoroNative;
import com.CodeBySonu.VoxSherpa.vietnamese.VietnameseKokoroVoice;

import java.util.ArrayList;
import java.util.HashMap;

/** Adds the bundled Vietnamese voice and diagnostics entry to VoxSherpa's TTS settings UI. */
public class VietnameseTtssettingsActivity extends TtssettingsActivity {
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        injectBundledVietnameseVoice();
        installDiagnosticsMenu();
    }

    private void injectBundledVietnameseVoice() {
        boolean nativeAvailable = VietnameseKokoroNative.isAvailable();
        boolean bundled = VietnameseKokoroEngine.isBundled(this);
        TtsDiagnostics.info(this, "settings", "vietnamese_voice_probe",
                "nativeAvailable=" + nativeAvailable + ", bundled=" + bundled
                        + ", loadError=" + VietnameseKokoroNative.loadError());

        if (!bundled || groupedLanguageList == null) return;

        for (HashMap<String, Object> group : groupedLanguageList) {
            Object name = group.get("language_name");
            if (name != null && "Vietnamese".equalsIgnoreCase(name.toString())) return;
        }

        HashMap<String, Object> voice = new HashMap<>();
        voice.put("voice_id", VietnameseKokoroVoice.DIEM_TRINH.androidVoiceName);
        voice.put("display_name", VietnameseKokoroVoice.DIEM_TRINH.displayName);
        voice.put("subtitle", "Bundled Kokoro Vietnamese • Offline • Stage 1");
        voice.put("is_kokoro", "true");
        voice.put("sample_url", "");
        voice.put("model_type", "kokoro_vi");
        voice.put("onnx_path", "bundled://kokoro_vi/kokoro_vi.onnx");
        voice.put("tokens_path", "bundled://kokoro_vi/config.json");
        voice.put("voices_bin_path", "bundled://kokoro_vi/voicepacks/diem_trinh.f32le");
        voice.put("speaker_id", "0");

        ArrayList<HashMap<String, Object>> voices = new ArrayList<>();
        voices.add(voice);

        HashMap<String, Object> group = new HashMap<>();
        group.put("language_name", "Vietnamese");
        group.put("is_expanded", "true");
        group.put("voices", voices);
        groupedLanguageList.add(0, group);

        int noModelId = getResources().getIdentifier("txt_no_model", "id", getPackageName());
        if (noModelId != 0) {
            TextView noModel = findViewById(noModelId);
            if (noModel != null) noModel.setVisibility(View.GONE);
        }

        int recyclerId = getResources().getIdentifier("recyclerview_voices", "id", getPackageName());
        if (recyclerId != 0) {
            androidx.recyclerview.widget.RecyclerView recycler = findViewById(recyclerId);
            if (recycler != null) {
                recycler.setVisibility(View.VISIBLE);
                if (recycler.getAdapter() != null) recycler.getAdapter().notifyDataSetChanged();
            }
        }
        TtsDiagnostics.info(this, "settings", "vietnamese_voice_added",
                "Added " + VietnameseKokoroVoice.DIEM_TRINH.androidVoiceName + " to settings UI.");
    }

    private void installDiagnosticsMenu() {
        int moreId = getResources().getIdentifier("img_more", "id", getPackageName());
        if (moreId == 0) return;
        View more = findViewById(moreId);
        if (more == null) return;
        more.setOnClickListener(view -> {
            PopupMenu popup = new PopupMenu(this, view);
            popup.getMenu().add(0, 1, 0, "System TTS Settings");
            popup.getMenu().add(0, 2, 1, "TTS Diagnostics");
            popup.setOnMenuItemClickListener(item -> {
                if (item.getItemId() == 1) {
                    try {
                        Intent intent = new Intent("com.android.settings.TTS_SETTINGS");
                        startActivity(intent);
                    } catch (Throwable t) {
                        TtsDiagnostics.error(this, "settings", "open_system_tts_failed", t.toString(), t);
                    }
                    return true;
                }
                if (item.getItemId() == 2) {
                    startActivity(new Intent(this, TtsDiagnosticsActivity.class));
                    return true;
                }
                return false;
            });
            popup.show();
        });
    }
}
