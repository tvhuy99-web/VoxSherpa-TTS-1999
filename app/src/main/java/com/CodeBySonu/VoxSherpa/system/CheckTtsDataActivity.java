package com.CodeBySonu.VoxSherpa.system;

import android.app.Activity;
import android.content.Intent;
import android.content.SharedPreferences;
import android.os.Bundle;
import android.speech.tts.TextToSpeech;

import com.CodeBySonu.VoxSherpa.KokoroVoiceHelper;
import com.CodeBySonu.VoxSherpa.vietnamese.VietnameseKokoroEngine;
import com.CodeBySonu.VoxSherpa.vietnamese.VietnameseKokoroNative;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.Set;

/** Reports both downloaded and bundled engine data to Android's TTS framework. */
public class CheckTtsDataActivity extends Activity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        ArrayList<String> availableVoices = new ArrayList<>();
        ArrayList<String> unavailableVoices = new ArrayList<>();

        try {
            // Register bundled Vietnamese on the exact same primary default path used by
            // downloaded voices before answering Android's CHECK_TTS_DATA request.
            TtsDefaultHelper.syncDefaultVoices(this);

            SharedPreferences sp = getSharedPreferences("sp1", MODE_PRIVATE);
            String allData = sp.getString("models_data", "[]");
            java.util.ArrayList<java.util.HashMap<String, Object>> downloadedModels =
                    new com.google.gson.Gson().fromJson(
                            allData,
                            new com.google.gson.reflect.TypeToken<java.util.ArrayList<java.util.HashMap<String, Object>>>() {}.getType());

            Set<String> uniqueLocales = new HashSet<>();

            if (downloadedModels != null) {
                boolean isKokoroDownloaded = false;
                for (java.util.HashMap<String, Object> m : downloadedModels) {
                    String onnxPath = m.containsKey("onnx_path") && m.get("onnx_path") != null
                            ? m.get("onnx_path").toString() : "";
                    if (onnxPath.isEmpty()) continue;

                    boolean isKokoroType = m.containsKey("type") && m.get("type") != null
                            && m.get("type").toString().contains("Kokoro");
                    if (isKokoroType) {
                        isKokoroDownloaded = true;
                    } else {
                        String rawLanguage = m.containsKey("language") && m.get("language") != null
                                ? m.get("language").toString() : "";
                        if (!rawLanguage.isEmpty()) {
                            String[] isoLang = TtsLocaleHelper.getTtsLanguageArray(rawLanguage);
                            if (isoLang != null && isoLang[0] != null && !isoLang[0].isEmpty()) {
                                String locale = isoLang[0];
                                if (isoLang.length > 1 && isoLang[1] != null && !isoLang[1].isEmpty()) {
                                    locale += "-" + isoLang[1];
                                }
                                uniqueLocales.add(locale);
                            }
                        }
                    }
                }

                if (isKokoroDownloaded) {
                    java.util.List<String> kokoroLangs = KokoroVoiceHelper.getAvailableLanguages();
                    for (String lang : kokoroLangs) {
                        String[] isoLang = TtsLocaleHelper.getTtsLanguageArray(lang);
                        if (isoLang != null && isoLang[0] != null && !isoLang[0].isEmpty()) {
                            String locale = isoLang[0];
                            if (isoLang.length > 1 && isoLang[1] != null && !isoLang[1].isEmpty()) {
                                locale += "-" + isoLang[1];
                            }
                            uniqueLocales.add(locale);
                        }
                    }
                }
            }

            boolean vietnameseBundled = VietnameseKokoroEngine.isBundled(this)
                    && VietnameseKokoroNative.isAvailable();
            if (vietnameseBundled) {
                // Android specifies lang-COUNTRY-variant for CHECK_TTS_DATA results.
                uniqueLocales.add("vie-VNM");
                // Start loading before TalkBack selects the engine. onLoadVoice/onLoadLanguage
                // are only hints and may arrive too late for the first utterance.
                VietnameseKokoroEngine.getInstance().prewarmAsync(this, "check_tts_data");
            }

            availableVoices.addAll(uniqueLocales);
            TtsDiagnostics.info(this, "check_tts_data", "result",
                    "available=" + availableVoices + ", vietnameseBundled=" + vietnameseBundled
                            + ", nativeAvailable=" + VietnameseKokoroNative.isAvailable()
                            + ", nativeError=" + VietnameseKokoroNative.loadError());

            Intent returnData = new Intent();
            returnData.putStringArrayListExtra(TextToSpeech.Engine.EXTRA_AVAILABLE_VOICES, availableVoices);
            returnData.putStringArrayListExtra(TextToSpeech.Engine.EXTRA_UNAVAILABLE_VOICES, unavailableVoices);
            setResult(TextToSpeech.Engine.CHECK_VOICE_DATA_PASS, returnData);
        } catch (Throwable t) {
            TtsDiagnostics.error(this, "check_tts_data", "exception", t.toString(), t);
            Intent fallback = new Intent();
            fallback.putStringArrayListExtra(TextToSpeech.Engine.EXTRA_AVAILABLE_VOICES, availableVoices);
            fallback.putStringArrayListExtra(TextToSpeech.Engine.EXTRA_UNAVAILABLE_VOICES, unavailableVoices);
            setResult(TextToSpeech.Engine.CHECK_VOICE_DATA_PASS, fallback);
        }

        finish();
    }
}