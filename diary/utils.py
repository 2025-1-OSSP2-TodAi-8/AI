import os, json, tempfile, numpy as np, librosa, soundfile as sf, openai
import torch, torch.nn as nn, torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from faster_whisper import WhisperModel

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# -------------------- 경로/장치 --------------------
BASE_DIR = os.path.dirname(os.path.realpath(__file__))
TEXT_DIR = os.path.join(BASE_DIR, "text")
AUDIO_STATE_PATH = os.path.join(BASE_DIR, "audio", "pytorch_model.pth")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
compute_type = "float16" if device.type == "cuda" else "int8"

# -------------------- Whisper --------------------
_whisper = WhisperModel("large-v3", device=device.type, compute_type=compute_type)

# -------------------- 텍스트 감정 모델 --------------------
try:
    _tok = AutoTokenizer.from_pretrained(TEXT_DIR)
    _text_model = (
        AutoModelForSequenceClassification.from_pretrained(TEXT_DIR).to(device).eval()
    )
except Exception as e:
    raise RuntimeError(f"[TEXT MODEL] 로드 실패: {e}. TEXT_DIR={TEXT_DIR}")

with open(os.path.join(TEXT_DIR, "label_map.json"), "r", encoding="utf-8") as f:
    _lm = json.load(f)
_id2label = {int(k): v for k, v in _lm["id2label"].items()}  # 0~5 → 한글 라벨
_KO_LABELS = [
    _id2label[i] for i in range(len(_id2label))
]  # ['기쁨','당황','분노','불안','상처','슬픔']
_NUM_LABELS_TEXT = len(_id2label)


def split_sents(text: str):
    import re

    return [s.strip() for s in re.split(r"[.?!\n]", text) if s.strip()]


def analyze_text_emotion(text: str, max_len=256):
    """문장 단위 inference → 퍼센트(dict, 0~100), total, details, 원문 반환"""
    sents = split_sents(text)
    if not sents:
        return {"percent": {}, "total": 0, "details": [], "text": text}

    counts = {lab: 0 for lab in _KO_LABELS}
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
    perc = {
        lab: (round((counts[lab] / total) * 100, 2) if total > 0 else 0.0)
        for lab in _KO_LABELS
    }
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

EMOTION_LABELS_EN = [
    "ANGRY",
    "SAD",
    "DISGUST",
    "HAPPY",
    "FEAR",
    "SURPRISE",
]  # 출력 순서(오디오)

# ---- Baseline (남/여) ----
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

BASE_LINE_MEAN_FEMALE = np.array(
    [
        -460.19843,
        46.570786,
        1.1123316,
        17.595436,
        -0.57482404,
        11.101548,
        -7.8497324,
        2.2953954,
        -2.1675904,
        -4.268735,
        -4.0286107,
        -5.2267838,
        -6.829859,
    ],
    dtype=np.float32,
)
BASE_LINE_STD_FEMALE = np.array(
    [
        25.068605,
        10.782973,
        6.916395,
        6.7128243,
        3.8709269,
        2.940329,
        3.7364106,
        2.884662,
        2.534034,
        2.1726286,
        2.3333669,
        2.8525667,
        2.571768,
    ],
    dtype=np.float32,
)


def _pick_baseline(gender: str):
    g = (gender or "MALE").strip().upper()
    if g == "FEMALE":
        return BASE_LINE_MEAN_FEMALE, BASE_LINE_STD_FEMALE
    # default MALE
    return BASE_LINE_MEAN_MALE, BASE_LINE_STD_MALE


# -------------------- 신호/전사 --------------------
def ensure_16k_mono(in_path: str) -> str:
    y, sr = librosa.load(in_path, sr=16000, mono=True)
    out_path = (
        in_path if in_path.endswith("_16k.wav") else in_path.replace(".wav", "_16k.wav")
    )
    sf.write(out_path, y, 16000)
    return out_path


def transcribe_whisper(wav_path: str, lang="ko") -> str:
    segments, _info = _whisper.transcribe(
        wav_path, language=lang, vad_filter=True, word_timestamps=True, beam_size=5
    )
    return "".join(seg.text for seg in segments)


def extract_sequence_features(wav_path: str, max_len=100):
    y, sr = librosa.load(wav_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).T
    if len(mfcc) < max_len:
        mfcc = np.pad(mfcc, ((0, max_len - len(mfcc)), (0, 0)), mode="constant")
    else:
        mfcc = mfcc[:max_len]
    return mfcc


def analyze_audio_emotion(wav_path: str, gender="MALE"):
    """오디오 확률(en) 및 Top-1"""
    mean_vec, std_vec = _pick_baseline(gender)
    features = extract_sequence_features(wav_path, max_len=100)  # (100,13)
    delta = (features - mean_vec) / (std_vec + 1e-8)
    x = (
        torch.from_numpy(delta[None].transpose(0, 2, 1)).float().to(device)
    )  # (1,13,100)
    with torch.no_grad():
        logits = _audio_model(x)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()  # (6,)
    top = int(np.argmax(probs))
    return {
        "probs_en": {lab: float(probs[i]) for i, lab in enumerate(EMOTION_LABELS_EN)},
        "top1_en": {"label": EMOTION_LABELS_EN[top], "prob": float(probs[top])},
        "vector_en": probs,  # numpy (6,)
    }


# -------------------- 라벨 매핑 & 융합 --------------------
# 권장 전이행렬 (열=KO: [기쁨, 당황, 분노, 불안, 상처, 슬픔], 행=EN: [ANGRY,SAD,DISGUST,HAPPY,FEAR,SURPRISE])
_M_KO2EN = np.array(
    [
        [0.0, 0.0, 1.0, 0.0, 0.2, 0.0],  # ANGRY   <- 분노(1.0), 상처(0.2)
        [0.0, 0.0, 0.0, 0.0, 0.6, 1.0],  # SAD     <- 슬픔(1.0), 상처(0.6)
        [0.0, 0.0, 0.0, 0.0, 0.2, 0.0],  # DISGUST <- 상처(0.2)
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # HAPPY   <- 기쁨(1.0)
        [0.0, 0.3, 0.0, 0.8, 0.0, 0.0],  # FEAR    <- 불안(0.8), 당황(0.3)
        [0.0, 0.7, 0.0, 0.2, 0.0, 0.0],  # SURPRISE<- 당황(0.7), 불안(0.2)
    ],
    dtype=np.float32,
)

# EN -> KO 근사 역매핑 (행=KO, 열=EN)  *명확한 역행렬이 아니므로 전문가적 휴리스틱
_M_EN2KO = np.array(
    [
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],  # 기쁨     <- HAPPY
        [
            0.0,
            0.0,
            0.0,
            0.2,
            0.2,
            0.7,
        ],  # 당황     <- SURPRISE(주), FEAR(부), HAPPY(소량)
        [0.8, 0.0, 0.3, 0.0, 0.0, 0.0],  # 분노     <- ANGRY(주), DISGUST(부)
        [0.0, 0.0, 0.0, 0.0, 0.8, 0.3],  # 불안     <- FEAR(주), SURPRISE(부)
        [0.2, 0.2, 0.7, 0.0, 0.0, 0.0],  # 상처     <- DISGUST(주), ANGRY/SAD(부)
        [0.0, 0.8, 0.0, 0.0, 0.0, 0.0],  # 슬픔     <- SAD(주)
    ],
    dtype=np.float32,
)


def _normalize(v):
    v = np.clip(np.asarray(v, dtype=np.float32), 1e-12, 1.0)
    s = float(v.sum())
    return v / s if s > 0 else np.full_like(v, 1.0 / len(v))


def _ko_percent_to_prob_vector(perc_dict):
    # KO 순서로 벡터 생성: ['기쁨','당황','분노','불안','상처','슬픔']
    vec = np.array(
        [float(perc_dict.get(k, 0.0)) / 100.0 for k in _KO_LABELS], dtype=np.float32
    )
    return _normalize(vec)


def fuse_text_audio(
    text_perc_ko: dict, audio_probs_en: np.ndarray, w_text=0.7, w_audio=0.3
):
    """텍스트(KO) -> EN으로 투영 후 오디오(EN)와 가중합 -> 최종 KO 확률 벡터 반환"""
    t_ko = _ko_percent_to_prob_vector(text_perc_ko)  # (6,)
    t_en = _normalize(_M_KO2EN @ t_ko)  # (6,) EN
    a_en = _normalize(audio_probs_en)  # (6,)
    fused_en = _normalize(w_text * t_en + w_audio * a_en)  # (6,) EN
    fused_ko = _normalize(_M_EN2KO @ fused_en)  # (6,) KO
    return fused_ko  # KO 순서


# -------------------- GPT 요약 --------------------
def empathetic_summary(text: str) -> str:
    """
    OPENAI_API_KEY 환경변수 필요.
    최신 SDK(OpenAI 1.x)와 구버전(openai 0.x) 모두 지원 시도.
    """
    prompt = (
        "다음 일기를 한국어로 1~2문장, 공감하는 톤으로 요약해 주세요. ex) ~하셨군요, ~하셨을 것 같아요"
        "상대방의 감정을 인정하며 과장하지 말고, 존댓말로 간결하게 작성하세요.\n\n"
        f"일기:\n{text}\n\n"
        "출력 형식: 한 문단의 자연스러운 문장 (따옴표 없이)"
    )

    try:

        openai.api_key = os.environ.get("OPENAI_API_KEY")
        resp = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "당신은 공감 능력이 뛰어난 한국어 요약가입니다.",
                },
                {"role": "user", "content": prompt},
            ],
        )
        return resp.choices[0].message["content"].strip()
    except Exception as e:
        # 실패 시 원문 일부라도 반환
        return f"(요약 실패) {e}"


# -------------------- 업로드 파일 파이프라인 --------------------
def run_pipeline_on_uploaded_file(django_file, gender="MALE", lang="ko"):
    """
    업로드된 파일을 임시 디렉토리에 저장 → 16k 변환 → STT → 텍스트/오디오 감정 → 융합 → 요약
    반환: { "summary": str, "emotion_analysis": [float x6] }  # KO 순서
    """
    with tempfile.TemporaryDirectory() as td:
        in_path = os.path.join(td, "input.wav")
        with open(in_path, "wb") as f:
            for chunk in django_file.chunks():
                f.write(chunk)

        p16 = ensure_16k_mono(in_path)

        # 1) STT
        text = transcribe_whisper(p16, lang=lang)

        # 2) 텍스트 감정 (KO 퍼센트 dict)
        text_result = analyze_text_emotion(text)
        text_perc_ko = text_result.get("percent", {})

        # 3) 오디오 감정 (EN probs)
        audio_result = analyze_audio_emotion(p16, gender=gender)
        audio_probs_en = audio_result["vector_en"]  # numpy (6,)

        # 4) 융합 (텍스트 0.7, 오디오 0.3)
        fused_ko = fuse_text_audio(
            text_perc_ko, audio_probs_en, w_text=0.7, w_audio=0.3
        )

        # 5) GPT 요약
        summary = empathetic_summary(text)

        # 최종 스펙 맞추기
        return {
            "summary": summary,
            # KO 라벨 순서 고정: ['기쁨','당황','분노','불안','상처','슬픔']
            "emotion_analysis": [float(x) for x in fused_ko.tolist()],
        }
