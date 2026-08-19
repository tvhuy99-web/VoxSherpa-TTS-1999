package com.CodeBySonu.VoxSherpa;

import android.content.Intent;
import android.os.Bundle;
import android.view.View;
import android.widget.PopupMenu;
import android.widget.TextView;
import android.widget.Toast;

import com.CodeBySonu.VoxSherpa.system.TtsDefaultHelper;
import com.CodeBySonu.VoxSherpa.system.TtsDiagnostics;
import com.CodeBySonu.VoxSherpa.system.TtsDiagnosticsActivity;
import com.CodeBySonu.VoxSherpa.vietnamese.VietnameseKokoroEngine;
import com.CodeBySonu.VoxSherpa.vietnamese.VietnameseKokoroNative;
import com.CodeBySonu.VoxSherpa.vietnamese.VietnameseKokoroVoice;

import java.util.ArrayList;
import java.util.HashMap;

/** Unified TTS settings screen: downloaded voices and bundled Vietnamese use the same selection path. */
public class VietnameseTtssettingsActivity extends TtssettingsActivity {
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        TtsDefaultHelper.syncDefaultVoices(this);
        VietnameseKokoroEngine.getInstance().prewarmAsync(this, "vietnamese_settings_opened");

        TtsDiagnostics.info(this, "settings", "vietnamese_settings_opened",
                "Unified Vietnamese-aware TTS settings screen opened; primary defaults synchronized and prewarm requested.");

        injectBundledVietnameseVoice();
        // Re-assert after layout/adapter work to make OEM timing or generated UI code harmless.
        View content = findViewById(android.R.id.content);
        if (content != null) {
            content.post(this::injectBundledVietnameseVoice);
            content.postDelayed(this::injectBundledVietnameseVoice, 250L);
        }
        installDiagnosticsMenu();
    }

    @SuppressWarnings("unchecked")
    private void injectBundledVietnameseVoice() {
        boolean nativeAvailable = VietnameseKokoroNative.isAvailable();
        boolean bundled = VietnameseKokoroEngine.isBundled(this);
        TtsDiagnostics.info(this, "settings", "vietnamese_voice_probe",
                "nativeAvailable=" + nativeAvailable + ", bundled=" + bundled
                        + ", groupedLanguageList=" + (groupedLanguageList != null)
                        + ", loadError=" + VietnameseKokoroNative.loadError());

        if (!bundled || !nativeAvailable) {
            TtsDiagnostics.warn(this, "settings", "vietnamese_voice_not_added",
                    "Bundled Vietnamese unavailable; nativeAvailable=" + nativeAvailable
                            + ", bundled=" + bundled);
            return;
        }

        if (groupedLanguageList == null) groupedLanguageList = new ArrayList<>();

        HashMap<String, Object> vietnameseGroup = null;
        for (HashMap<String, Object> group : groupedLanguageList) {
            Object name = group.get("language_name");
            if (name == null) continue;
            String value = name.toString().trim();
            if ("Vietnamese".equalsIgnoreCase(value)
                    || "Tiếng Việt".equalsIgnoreCase(value)
                    || value.toLowerCase(java.util.Locale.ROOT).contains("vietnamese")) {
                vietnameseGroup = group;
                break;
            }
        }

        ArrayList<HashMap<String, Object>> voices;
        if (vietnameseGroup == null) {
            vietnameseGroup = new HashMap<>();
            vietnameseGroup.put("language_name", "Vietnamese");
            vietnameseGroup.put("is_expanded", "true");
            voices = new ArrayList<>();
            vietnameseGroup.put("voices", voices);
            groupedLanguageList.add(0, vietnameseGroup);
            TtsDiagnostics.info(this, "settings", "vietnamese_group_added",
                    "Created first-class Installed Languages group for bundled Vietnamese.");
        } else {
            Object existingVoices = vietnameseGroup.get("voices");
            if (existingVoices instanceof ArrayList) {
                voices = (ArrayList<HashMap<String, Object>>) existingVoices;
            } else {
                voices = new ArrayList<>();
                vietnameseGroup.put("voices", voices);
            }
            vietnameseGroup.put("is_expanded", "true");
        }

        HashMap<String, Object> voice = null;
        for (HashMap<String, Object> existing : voices) {
            Object id = existing.get("voice_id");
            if (id != null && VietnameseKokoroVoice.DIEM_TRINH.androidVoiceName.equals(id.toString())) {
                voice = existing;
                break;
            }
        }

        if (voice == null) {
            voice = new HashMap<>();
            voices.add(0, voice);
            TtsDiagnostics.info(this, "settings", "vietnamese_voice_added",
                    "Added bundled Diem Trinh to the primary Installed Languages list.");
        }

        // Use exactly the same keys consumed by the existing TTS settings adapter/sp5 path.
        voice.put("voice_id", VietnameseKokoroVoice.DIEM_TRINH.androidVoiceName);
        voice.put("display_name", VietnameseKokoroVoice.DIEM_TRINH.displayName);
        voice.put("subtitle", "Bundled Kokoro Vietnamese • Offline • Installed");
        voice.put("is_kokoro", "true");
        voice.put("sample_url", "");
        voice.put("model_type", "kokoro_vi");
        voice.put("onnx_path", "bundled://kokoro_vi/kokoro_vi.onnx");
        voice.put("tokens_path", "bundled://kokoro_vi/config.json");
        voice.put("voices_bin_path", "bundled://kokoro_vi/voicepacks/diem_trinh.f32le");
        voice.put("speaker_id", "0");

        getSharedPreferences("sp1", MODE_PRIVATE).edit()
                .putString("default_voice_Vietnamese", VietnameseKokoroVoice.DIEM_TRINH.androidVoiceName)
                .apply();
        TtsDefaultHelper.syncDefaultVoices(this);

        refreshVoiceListUi();
        TtsDiagnostics.info(this, "settings", "vietnamese_voice_visible",
                "Unified list now contains Vietnamese group with voiceCount=" + voices.size());
    }

    private void refreshVoiceListUi() {
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
                TtsDiagnostics.info(this, "settings", "voice_list_refreshed",
                        "adapter=" + (recycler.getAdapter() != null)
                                + ", groups=" + (groupedLanguageList == null ? -1 : groupedLanguageList.size()));
            }
        }
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
                    openSystemTtsAfterWarmup();
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

    private void openSystemTtsAfterWarmup() {
        VietnameseKokoroEngine engine = VietnameseKokoroEngine.getInstance();
        if (engine.isReady()) {
            openSystemTtsSettings();
            return;
        }

        Toast.makeText(this,
                "Preparing Vietnamese voice for TalkBack… System TTS will open when ready.",
                Toast.LENGTH_LONG).show();
        TtsDiagnostics.info(this, "settings", "system_tts_waiting_for_warmup",
                "Delaying System TTS navigation until bundled Vietnamese is warm to avoid first TalkBack silence.");
        engine.prewarmAsync(this, "before_system_tts_settings");

        new Thread(() -> {
            long deadline = android.os.SystemClock.elapsedRealtime() + 35000L;
            while (!engine.isReady() && android.os.SystemClock.elapsedRealtime() < deadline) {
                try {
                    Thread.sleep(100L);
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                    break;
                }
            }
            boolean ready = engine.isReady();
            runOnUiThread(() -> {
                TtsDiagnostics.info(this, "settings", "system_tts_warmup_gate_finished",
                        "ready=" + ready + ", " + engine.performanceState(this));
                openSystemTtsSettings();
            });
        }, "KokoroVi-SystemTtsGate").start();
    }

    private void openSystemTtsSettings() {
        try {
            Intent intent = new Intent("com.android.settings.TTS_SETTINGS");
            startActivity(intent);
        } catch (Throwable t) {
            TtsDiagnostics.error(this, "settings", "open_system_tts_failed", t.toString(), t);
            Toast.makeText(this, "Unable to open System TTS Settings.", Toast.LENGTH_SHORT).show();
        }
    }
}