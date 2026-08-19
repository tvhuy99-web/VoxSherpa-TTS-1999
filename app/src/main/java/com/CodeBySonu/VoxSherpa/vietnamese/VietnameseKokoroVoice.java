package com.CodeBySonu.VoxSherpa.vietnamese;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Locale;

/** Canonical registry for every bundled Kokoro Vietnamese voice. */
public final class VietnameseKokoroVoice {
    public static final Locale LOCALE = new Locale("vi", "VN");

    public static final VietnameseKokoroVoice DIEM_TRINH = voice("diem_trinh", "Diễm Trinh");
    public static final VietnameseKokoroVoice HUNG_THINH = voice("hung_thinh", "Hưng Thịnh");
    public static final VietnameseKokoroVoice MAI_LINH = voice("mai_linh", "Mai Linh");
    public static final VietnameseKokoroVoice MAI_LOAN = voice("mai_loan", "Mai Loan");
    public static final VietnameseKokoroVoice MANH_DUNG = voice("manh_dung", "Mạnh Dũng");
    public static final VietnameseKokoroVoice MY_YEN = voice("my_yen", "Mỹ Yến");
    public static final VietnameseKokoroVoice NGOC_HUYEN = voice("ngoc_huyen", "Ngọc Huyền");
    public static final VietnameseKokoroVoice PHAT_TAI = voice("phat_tai", "Phát Tài");
    public static final VietnameseKokoroVoice THANH_DAT = voice("thanh_dat", "Thành Đạt");
    public static final VietnameseKokoroVoice THUC_TRINH = voice("thuc_trinh", "Thục Trinh");
    public static final VietnameseKokoroVoice TUAN_NGOC = voice("tuan_ngoc", "Tuấn Ngọc");
    public static final VietnameseKokoroVoice STORYVERT = voice("storyvert", "Storyvert");
    public static final VietnameseKokoroVoice DUC_AN = voice("duc_an", "Đức An");
    public static final VietnameseKokoroVoice DUC_DUY = voice("duc_duy", "Đức Duy");

    public static final VietnameseKokoroVoice DEFAULT = DIEM_TRINH;

    private static final List<VietnameseKokoroVoice> ALL = Collections.unmodifiableList(Arrays.asList(
            DIEM_TRINH,
            HUNG_THINH,
            MAI_LINH,
            MAI_LOAN,
            MANH_DUNG,
            MY_YEN,
            NGOC_HUYEN,
            PHAT_TAI,
            THANH_DAT,
            THUC_TRINH,
            TUAN_NGOC,
            STORYVERT,
            DUC_AN,
            DUC_DUY
    ));

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

    private static VietnameseKokoroVoice voice(String id, String displayName) {
        return new VietnameseKokoroVoice(id, displayName, "kokoro_vi/voicepacks/" + id + ".f32le");
    }

    public static List<VietnameseKokoroVoice> all() {
        return ALL;
    }

    public static VietnameseKokoroVoice fromId(String id) {
        if (id == null) return null;
        String normalized = id.trim();
        for (VietnameseKokoroVoice voice : ALL) {
            if (voice.id.equals(normalized)) return voice;
        }
        return null;
    }

    public static VietnameseKokoroVoice fromAndroidVoiceName(String name) {
        if (name == null) return null;
        String normalized = name.trim();
        for (VietnameseKokoroVoice voice : ALL) {
            if (voice.androidVoiceName.equals(normalized) || voice.id.equals(normalized)) return voice;
        }
        return null;
    }

    @Override
    public String toString() {
        return displayName;
    }
}
