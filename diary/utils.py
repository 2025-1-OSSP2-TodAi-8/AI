import os
import json
import tempfile
import numpy as np
import librosa
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from faster_whisper import WhisperModel

# -------------------- 경로 설정 --------------------
BASE_DIR = os.path.dirname(os.path.realpath(__file__))
TEXT_DIR = os.path.join(BASE_DIR, "text")  # 학습된 텍스트 감정 모델 디렉토리
AUDIO_STATE_PATH = os.path.join(BASE_DIR, "audio", "pytorch_model.pth")

# -------------------- 디바이스/Whisper 세팅 --------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# CPU에서는 int8 추천, GPU면 float16 권장
compute_type = "float16" if device.type == "cuda" else "int8"

# 전역 싱글톤처럼 한번만 로드
_whisper = WhisperModel("large-v3", device=device.type, compute_type=compute_type)

# -------------------- 텍스트 감정 모델 로딩 --------------------
try:
    _tok = AutoTokenizer.from_pretrained(TEXT_DIR)
    _text_model = (
        AutoModelForSequenceClassification.from_pretrained(TEXT_DIR).to(device).eval()
    )
except Exception as e:
    raise RuntimeError(f"[TEXT MODEL] 로드 실패: {e}. TEXT_DIR={TEXT_DIR}")

with open(os.path.join(TEXT_DIR, "label_map.json"), "r", encoding="utf-8") as f:
    _lm = json.load(f)
_id2label = {int(k): v for k, v in _lm["id2label"].items()}
_NUM_LABELS_TEXT = len(_id2label)


def split_sents(text: str):
    import re

    return [s.strip() for s in re.split(r"[.?!\n]", text) if s.strip()]


def analyze_text_emotion(text: str, max_len=256):
    """
    문장 단위로 inference → 퍼센트 결과(dict)와 총합/문장별 라벨 반환
    """
    sents = split_sents(text)
    if not sents:
        return {"percent": {}, "total": 0, "details": [], "text": text}

    counts = {_id2label[i]: 0 for i in range(_NUM_LABELS_TEXT)}
    details = []

    with torch.no_grad():
        for s in sents:
            enc = _tok(
                s,
                truncation=True,
                padding=True,
                max_length=max_len,
                return_tensors="pt",
            ).to(device)
            logits = _text_model(**enc).logits
            pred = int(logits.argmax(-1).item())
            lab = _id2label[pred]
            counts[lab] += 1
            details.append((s, lab))

    total = sum(counts.values())
    if total == 0:
        perc = {lab: 0.0 for lab in counts.keys()}
    else:
        perc = {lab: round((cnt / total) * 100, 2) for lab, cnt in counts.items()}

    return {"percent": perc, "total": total, "details": details, "text": text}


# -------------------- 오디오 감정 모델 --------------------
class PyTorchAudioModel(nn.Module):
    def __init__(self, num_labels=6):
        super().__init__()
        self.conv1 = nn.Conv1d(13, 64, kernel_size=5, padding="same")
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(2, 2)

        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding="same")
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(2, 2)

        self.bilstm = nn.LSTM(128, 64, bidirectional=True, batch_first=True)

        self.dense1 = nn.Linear(128, 128)
        self.dense2 = nn.Linear(128, num_labels)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = x.permute(0, 2, 1)  # (B,C,L)->(B,L,C)
        _, (h_n, _) = self.bilstm(x)
        x = torch.cat([h_n[-2], h_n[-1]], dim=1)
        x = F.relu(self.dense1(x))
        return self.dense2(x)


_audio_model = PyTorchAudioModel(num_labels=6).to(device)
_audio_model.load_state_dict(torch.load(AUDIO_STATE_PATH, map_location=device))
_audio_model.eval()

EMOTION_LABELS_EN = ["ANGRY", "SAD", "DISGUST", "HAPPY", "FEAR", "SURPRISE"]

# 베이스라인 (남성 예시) — 필요 시 설정에서 주입 가능
BASE_LINE_MEAN_MALE = np.array(
    [
        -488.7764,
        75.9438,
        6.369161,
        21.888578,
        5.252565,
        12.948459,
        -2.9029474,
        2.1715217,
        6.144363,
        2.0456758,
        -4.0672646,
        1.4047805,
        -0.85426885,
    ],
    dtype=np.float32,
)
BASE_LINE_STD_MALE = np.array(
    [
        11.675788,
        12.669461,
        5.7222886,
        4.909043,
        4.4742537,
        5.8538017,
        3.2380152,
        4.0887833,
        1.9294198,
        2.4097118,
        2.7102668,
        1.6911668,
        1.7729696,
    ],
    dtype=np.float32,
)


def ensure_16k_mono(in_path: str) -> str:
    """입력 파일을 16kHz mono로 변환한 wav 경로 반환(같은 디렉토리에 *_16k.wav 생성)"""
    y, sr = librosa.load(in_path, sr=16000, mono=True)
    out_path = (
        in_path if in_path.endswith("_16k.wav") else in_path.replace(".wav", "_16k.wav")
    )
    sf.write(out_path, y, 16000)
    return out_path


def transcribe_whisper(wav_path: str, lang="ko") -> str:
    """Whisper로 텍스트 추출"""
    segments, info = _whisper.transcribe(
        wav_path,
        language=lang,
        vad_filter=True,
        word_timestamps=True,
        beam_size=5,
    )
    return "".join(seg.text for seg in segments)


def extract_sequence_features(wav_path: str, max_len=100):
    """MFCC 13차 추출 후 (max_len, 13)로 패딩/자르기"""
    y, sr = librosa.load(wav_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).T
    if len(mfcc) < max_len:
        pad = max_len - len(mfcc)
        mfcc = np.pad(mfcc, ((0, pad), (0, 0)), mode="constant")
    else:
        mfcc = mfcc[:max_len]
    return mfcc


def analyze_audio_emotion(wav_path: str):
    """오디오 기반 감정 확률 및 Top-1 반환"""
    features = extract_sequence_features(wav_path, max_len=100)  # (100,13)
    eps = 1e-8
    delta = (features - BASE_LINE_MEAN_MALE) / (BASE_LINE_STD_MALE + eps)
    x = (
        torch.from_numpy(delta[None].transpose(0, 2, 1)).float().to(device)
    )  # (1,13,100)

    with torch.no_grad():
        logits = _audio_model(x)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

    top_idx = int(np.argmax(probs))
    return {
        "probs": {lab: float(probs[i]) for i, lab in enumerate(EMOTION_LABELS_EN)},
        "top1": {"label": EMOTION_LABELS_EN[top_idx], "prob": float(probs[top_idx])},
    }


def run_pipeline_on_uploaded_file(django_file, lang="ko"):
    """
    업로드된 파일을 임시 디렉토리에 저장→16k 변환→STT→텍스트/오디오 감정 분석→임시파일 자동 삭제
    """
    with tempfile.TemporaryDirectory() as td:
        in_path = os.path.join(td, "input.wav")
        # chunk 단위 저장
        with open(in_path, "wb") as f:
            for chunk in django_file.chunks():
                f.write(chunk)

        # 16k 변환
        p16 = ensure_16k_mono(in_path)

        # 1) STT
        text = transcribe_whisper(p16, lang=lang)

        # 2) 텍스트 감정
        text_result = analyze_text_emotion(text)

        # 3) 오디오 감정
        audio_result = analyze_audio_emotion(p16)

        # TemporaryDirectory 컨텍스트를 벗어나면 파일 자동 삭제
        return {
            "stt_text": text,
            "text_emotion": text_result,  # {"percent": {...}, "total": int, "details": [...], "text": str}
            "audio_emotion": audio_result,  # {"probs": {...}, "top1": {...}}
        }
