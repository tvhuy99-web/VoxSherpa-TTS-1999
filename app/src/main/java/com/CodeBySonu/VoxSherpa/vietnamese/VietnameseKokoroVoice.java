package com.CodeBySonu.VoxSherpa.vietnamese;

import java.util.Locale;

/** Stage-1 voice registry. Phase 2 will expand this to all 14 bundled Vietnamese voices. */
public final class VietnameseKokoroVoice {
    public static final Locale LOCALE = new Locale("vi", "VN");
    public static final VietnameseKokoroVoice DIEM_TRINH = new VietnameseKokoroVoice(
            "diem_trinh",
            "Diễm Trinh",
            "kokoro_vi/voicepacks/diem_trinh.f32le"
    );

    public final String id;
    public final String displayName;
    public final String assetPath;
    public final String androidVoiceName;

    private VietnameseKokoroVoice(String id, String displayName, String assetPath) {
        this.id = id;
        this.displayName = displayName;
        this.assetPath = assetPath;
        this.androidVoiceName = "VoxSherpa_vi_" + id;
    }

    public static VietnameseKokoroVoice fromAndroidVoiceName(String name) {
        if (name == null) return null;
        if (DIEM_TRINH.androidVoiceName.equals(name) || DIEM_TRINH.id.equals(name)) return DIEM_TRINH;
        return null;
    }
}
