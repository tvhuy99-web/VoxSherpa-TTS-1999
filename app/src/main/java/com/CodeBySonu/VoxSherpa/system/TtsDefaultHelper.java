package com.CodeBySonu.VoxSherpa.system;

import android.content.Context;
import android.content.Intent;
import android.content.SharedPreferences;
import android.speech.tts.TextToSpeech;

import com.CodeBySonu.VoxSherpa.vietnamese.VietnameseKokoroEngine;
import com.CodeBySonu.VoxSherpa.vietnamese.VietnameseKokoroNative;
import com.CodeBySonu.VoxSherpa.vietnamese.VietnameseKokoroVoice;

/** Keeps the app's primary System-TTS defaults in sync with both downloaded and bundled voices. */
public class TtsDefaultHelper {

    private static final String VI_LANGUAGE_NAME = "Vietnamese";
    private static final String VI_DEFAULT_KEY = "default_voice_" + VI_LANGUAGE_NAME;
    private static final String VI_SYS_KEY = "sys_tts_" + VI_LANGUAGE_NAME;
    /** Bump when the bundled voice inventory changes so Android refreshes its cached voice list. */
    private static final String VI_ANNOUNCED_KEY = "kokoro_vi_data_announced_v3_14voices";

    public static void syncDefaultVoices(Context context) {
        if (context == null) return;

        Context app = context.getApplicationContext();
        SharedPreferences sp1 = app.getSharedPreferences("sp1", Context.MODE_PRIVATE);
        SharedPreferences sp5 = app.getSharedPreferences("sp5", Context.MODE_PRIVATE);
        SharedPreferences perf = app.getSharedPreferences("kokoro_vi_perf", Context.MODE_PRIVATE);

        boolean bundledVietnamese = VietnameseKokoroEngine.isBundled(app)
                && VietnameseKokoroNative.isAvailable();
        boolean dataChanged = false;
        boolean announceVietnamese = false;

        try {
            if (!perf.contains("cpu_threads")) {
                perf.edit().putInt("cpu_threads", 4).apply();
                TtsDiagnostics.info(app, "defaults", "cpu_default_initialized",
                        "Initialized bundled Vietnamese ORT default to 4 threads; benchmark may override it.");
            }

            SharedPreferences.Editor sp1Editor = sp1.edit();
            SharedPreferences.Editor sp5Editor = sp5.edit();

            if (bundledVietnamese) {
                String selectedViVoiceName = sp1.getString(VI_DEFAULT_KEY, "");
                VietnameseKokoroVoice selectedBundledVoice =
                        VietnameseKokoroVoice.fromAndroidVoiceName(selectedViVoiceName);
                if (selectedViVoiceName.isEmpty()) {
                    selectedBundledVoice = VietnameseKokoroVoice.DEFAULT;
                    selectedViVoiceName = selectedBundledVoice.androidVoiceName;
                    sp1Editor.putString(VI_DEFAULT_KEY, selectedViVoiceName);
                    dataChanged = true;
                }

                // If a bundled Vietnamese voice is selected, keep sp5 on the same first-class
                // path used by downloaded models. Do not overwrite a manually-installed voice.
                if (selectedBundledVoice != null) {
                    String existingVi = sp5.getString(VI_SYS_KEY, "");
                    boolean alreadySelected = false;
                    if (!existingVi.isEmpty()) {
                        try {
                            org.json.JSONObject existing = new org.json.JSONObject(existingVi);
                            alreadySelected = "kokoro_vi".equals(existing.optString("model_type", ""))
                                    && selectedBundledVoice.androidVoiceName.equals(
                                            existing.optString("voice_name", ""));
                        } catch (Throwable ignored) {}
                    }
                    if (!alreadySelected) {
                        int voiceIndex = VietnameseKokoroVoice.all().indexOf(selectedBundledVoice);
                        org.json.JSONObject sysJson = new org.json.JSONObject();
                        sysJson.put("model_type", "kokoro_vi");
                        sysJson.put("voice_name", selectedBundledVoice.androidVoiceName);
                        sysJson.put("onnx_path", "bundled://kokoro_vi/kokoro_vi.onnx");
                        sysJson.put("tokens_path", "bundled://kokoro_vi/config.json");
                        sysJson.put("voices_bin_path", "bundled://kokoro_vi/voicepacks/"
                                + selectedBundledVoice.id + ".f32le");
                        sysJson.put("speaker_id", Integer.toString(Math.max(0, voiceIndex)));
                        sp5Editor.putString(VI_SYS_KEY, sysJson.toString());
                        dataChanged = true;
                        TtsDiagnostics.info(app, "defaults", "bundled_vi_registered",
                                "Registered bundled Vietnamese default voice=" + selectedBundledVoice.id
                                        + " in the shared sp1/sp5 System-TTS path; totalVoices="
                                        + VietnameseKokoroVoice.all().size());
                    }
                }

                if (!sp5.getBoolean(VI_ANNOUNCED_KEY, false)) {
                    sp5Editor.putBoolean(VI_ANNOUNCED_KEY, true);
                    announceVietnamese = true;
                    dataChanged = true;
                }
            }

            String allData = sp1.getString("models_data", "[]");
            if (allData == null || allData.trim().isEmpty()) allData = "[]";

            java.util.ArrayList<java.util.HashMap<String, Object>> downloadedModels =
                    new com.google.gson.Gson().fromJson(
                            allData,
                            new com.google.gson.reflect.TypeToken<java.util.ArrayList<java.util.HashMap<String, Object>>>() {}.getType());

            if (downloadedModels != null && !downloadedModels.isEmpty()) {
                boolean isKokoroDownloaded = false;
                String globalKokoroOnnx = "";
                String globalKokoroTokens = "";
                String globalKokoroVoices = "";
                java.util.HashMap<String, java.util.HashMap<String, Object>> firstVitsPerLang =
                        new java.util.HashMap<>();

                for (java.util.HashMap<String, Object> m : downloadedModels) {
                    String onnxPath = m.containsKey("onnx_path") && m.get("onnx_path") != null
                            ? m.get("onnx_path").toString() : "";
                    if (onnxPath.isEmpty()) continue;

                    boolean isKokoroType = m.containsKey("type")
                            && m.get("type") != null
                            && m.get("type").toString().contains("Kokoro");
                    if (isKokoroType) {
                        isKokoroDownloaded = true;
                        globalKokoroOnnx = onnxPath;
                        globalKokoroTokens = m.containsKey("tokens_path") && m.get("tokens_path") != null
                                ? m.get("tokens_path").toString() : "";
                        globalKokoroVoices = m.containsKey("voices_bin_path") && m.get("voices_bin_path") != null
                                ? m.get("voices_bin_path").toString() : "";
                    } else {
                        String lang = m.containsKey("language") && m.get("language") != null
                                ? m.get("language").toString() : "";
                        if (!lang.isEmpty() && !firstVitsPerLang.containsKey(lang)) {
                            firstVitsPerLang.put(lang, m);
                        }
                    }
                }

                if (isKokoroDownloaded) {
                    java.util.List<String> kokoroLangs =
                            com.CodeBySonu.VoxSherpa.KokoroVoiceHelper.getAvailableLanguages();
                    java.util.List<com.CodeBySonu.VoxSherpa.KokoroVoiceHelper.VoiceItem> allKVoices =
                            com.CodeBySonu.VoxSherpa.KokoroVoiceHelper.getAllVoices();

                    for (String lang : kokoroLangs) {
                        String existingDefault = sp5.getString("sys_tts_" + lang, "");
                        if (!existingDefault.isEmpty()) continue;
                        for (com.CodeBySonu.VoxSherpa.KokoroVoiceHelper.VoiceItem kv : allKVoices) {
                            if (!kv.language.equals(lang)) continue;
                            org.json.JSONObject sysJson = new org.json.JSONObject();
                            sysJson.put("model_type", "kokoro");
                            sysJson.put("onnx_path", globalKokoroOnnx);
                            sysJson.put("tokens_path", globalKokoroTokens);
                            sysJson.put("voices_bin_path", globalKokoroVoices);
                            sysJson.put("speaker_id", String.valueOf(kv.speakerId));
                            sp5Editor.putString("sys_tts_" + lang, sysJson.toString());
                            dataChanged = true;
                            break;
                        }
                    }
                }

                for (String lang : firstVitsPerLang.keySet()) {
                    String existingDefault = sp5.getString("sys_tts_" + lang, "");
                    if (!existingDefault.isEmpty()) continue;

                    java.util.HashMap<String, Object> vits = firstVitsPerLang.get(lang);
                    String tk = "";
                    if (vits.containsKey("tokens_path") && vits.get("tokens_path") != null) {
                        tk = vits.get("tokens_path").toString();
                    } else if (vits.containsKey("lexicon_path") && vits.get("lexicon_path") != null) {
                        tk = vits.get("lexicon_path").toString();
                    } else if (vits.containsKey("config_path") && vits.get("config_path") != null) {
                        tk = vits.get("config_path").toString();
                    }

                    org.json.JSONObject sysJson = new org.json.JSONObject();
                    sysJson.put("model_type", "vits");
                    sysJson.put("onnx_path", vits.containsKey("onnx_path")
                            ? vits.get("onnx_path").toString() : "");
                    sysJson.put("tokens_path", tk);
                    sysJson.put("voices_bin_path", "");
                    sysJson.put("speaker_id", "-1");
                    sp5Editor.putString("sys_tts_" + lang, sysJson.toString());
                    dataChanged = true;
                }
            }

            if (dataChanged) {
                sp1Editor.apply();
                sp5Editor.apply();
            }

            if (announceVietnamese) {
                app.sendBroadcast(new Intent(TextToSpeech.Engine.ACTION_TTS_DATA_INSTALLED));
                TtsDiagnostics.info(app, "defaults", "tts_data_announced",
                        "Announced bundled Vietnamese TTS data globally; voiceCount="
                                + VietnameseKokoroVoice.all().size());
            }
        } catch (Throwable t) {
            TtsDiagnostics.error(app, "defaults", "sync_failed", t.toString(), t);
        }
    }
}
