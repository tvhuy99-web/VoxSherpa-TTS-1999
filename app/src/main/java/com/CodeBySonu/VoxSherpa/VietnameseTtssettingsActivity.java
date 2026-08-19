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

/** Unified TTS settings screen: downloaded voices and all bundled Vietnamese voices share one path. */
public class VietnameseTtssettingsActivity extends TtssettingsActivity {
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        TtsDefaultHelper.syncDefaultVoices(this);
        VietnameseKokoroEngine.getInstance().prewarmAsync(this, "vietnamese_settings_opened");

        TtsDiagnostics.info(this, "settings", "vietnamese_settings_opened",
                "Unified Vietnamese-aware TTS settings screen opened; bundledVoices="
                        + VietnameseKokoroVoice.all().size());

        injectBundledVietnameseVoices();
        View content = findViewById(android.R.id.content);
        if (content != null) {
            content.post(this::injectBundledVietnameseVoices);
            content.postDelayed(this::injectBundledVietnameseVoices, 250L);
        }
        installDiagnosticsMenu();
    }

    @SuppressWarnings("unchecked")
    private void injectBundledVietnameseVoices() {
        boolean nativeAvailable = VietnameseKokoroNative.isAvailable();
        boolean bundled = VietnameseKokoroEngine.isBundled(this);
        TtsDiagnostics.info(this, "settings", "vietnamese_voice_probe",
                "nativeAvailable=" + nativeAvailable + ", bundled=" + bundled
                        + ", expectedVoices=" + VietnameseKokoroVoice.all().size()
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
                    "Created Installed Languages group for bundled Vietnamese.");
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

        int added = 0;
        for (int registryIndex = 0; registryIndex < VietnameseKokoroVoice.all().size(); registryIndex++) {
            VietnameseKokoroVoice registryVoice = VietnameseKokoroVoice.all().get(registryIndex);
            HashMap<String, Object> voiceMap = null;
            for (HashMap<String, Object> existing : voices) {
                Object id = existing.get("voice_id");
                if (id != null && registryVoice.androidVoiceName.equals(id.toString())) {
                    voiceMap = existing;
                    break;
                }
            }
            if (voiceMap == null) {
                voiceMap = new HashMap<>();
                voices.add(voiceMap);
                added++;
            }

            voiceMap.put("voice_id", registryVoice.androidVoiceName);
            voiceMap.put("display_name", registryVoice.displayName);
            voiceMap.put("subtitle", "Bundled Kokoro Vietnamese • Offline • Installed");
            voiceMap.put("is_kokoro", "true");
            voiceMap.put("sample_url", "");
            voiceMap.put("model_type", "kokoro_vi");
            voiceMap.put("onnx_path", "bundled://kokoro_vi/kokoro_vi.onnx");
            voiceMap.put("tokens_path", "bundled://kokoro_vi/config.json");
            voiceMap.put("voices_bin_path", "bundled://kokoro_vi/voicepacks/" + registryVoice.id + ".f32le");
            voiceMap.put("speaker_id", Integer.toString(registryIndex));
        }

        TtsDefaultHelper.syncDefaultVoices(this);
        refreshVoiceListUi();
        TtsDiagnostics.info(this, "settings", "vietnamese_voice_visible",
                "Unified list contains Vietnamese voiceCount=" + voices.size()
                        + ", registryCount=" + VietnameseKokoroVoice.all().size()
                        + ", newlyAdded=" + added);
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
