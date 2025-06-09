# emotions/utils.py

import os
import torch
import torch.nn as nn
import numpy as np
import soundfile as sf
import librosa
import openai

from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Wav2Vec2Processor,
    Wav2Vec2ForSequenceClassification,
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

from kobert_transformers import get_tokenizer

wav2vec2_labels = ["Angry", "Sadness", "Disgust", "Happiness", "Fear", "Surprise"]

kobert_labels = ["Happiness", "Sadness", "Angry", "Surprise", "Fear", "Disgust"]


MIX_MATRIX = [
    # "기쁨", "슬픔", "분노", "놀람", "공포", "혐오"
    [0.444, 0.222, 0.0, 0.333, 0.0, 0.0],  # 기쁨
    [0.111, 0.222, 0.167, 0.167, 0.222, 0.111],  # 슬픔
    [0.0, 0.1875, 0.1875, 0.125, 0.25, 0.25],  # 분노
    [0.176, 0.176, 0.118, 0.176, 0.235, 0.118],  # 놀람
    [0.0, 0.210, 0.210, 0.210, 0.210, 0.158],  # 공포
    [0.0, 0.118, 0.235, 0.118, 0.176, 0.235],  # 혐오
]

EMOTION_KR_TO_EN = {
    "기쁨": "Happiness",
    "슬픔": "Sadness",
    "분노": "Angry",
    "놀람": "Surprise",
    "공포": "Fear",
    "혐오": "Disgust",
}

EMOTION_EN_TO_INDEX = {
    "Happiness": 0,
    "Sadness": 1,
    "Angry": 2,
    "Surprise": 3,
    "Fear": 4,
    "Disgust": 5,
}
openai.api_key = os.getenv("OPENAI_API_KEY")


def summarize(text: str) -> str:
    prompt = (
        """한 사람이 작성한 일기를 제공할꺼야, 다음 형식을 맞춰서 요약해,
            오늘은 (인간의 6가지 감정 분노,슬픔,혐오,행복,두려움,놀람 중 1개를 예측)을 느낀 하루 이셨군요
            (텍스트에서 찾은 원인)하셔서
            (1줄에서 판단한 감정)하게 느꼈던 날이었군요 ~하셨을 것 같아요.
            어떤 감정을 느끼는지에 중점을 두어 공감식으로 요약해:\n\n"""
        f"{text}\n\n"
        "요약:"
    )
    resp = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=150,
        n=1,
    )
    return resp.choices[0].message.content.strip()


# 장치 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Whisper 모델 로드
model_id = "openai/whisper-base"
processor = WhisperProcessor.from_pretrained(model_id)

whisper_model = WhisperForConditionalGeneration.from_pretrained(model_id).to(device)

whisper_model.eval()


checkpoint_path = "/Users/jaehyuk/Desktop/projects/TodAi/diary/emotion-text"

kmodel = AutoModelForSequenceClassification.from_pretrained(
    checkpoint_path,
    num_labels=6,
    problem_type="multi_label_classification",
    local_files_only=True,
)
tokenizer = AutoTokenizer.from_pretrained("monologg/kobert", trust_remote_code=True)


# Wav2Vec2 기반 음성 감정 분류기 로드
SAVED_W2VEMO_DIR = os.path.join(os.path.dirname(__file__), "emotion-audio")
wav2vec2_processor = Wav2Vec2Processor.from_pretrained(
    SAVED_W2VEMO_DIR, local_files_only=True
)
wav2vec2_emotion_model = Wav2Vec2ForSequenceClassification.from_pretrained(
    SAVED_W2VEMO_DIR,
    local_files_only=True,
    num_labels=6,
    problem_type="multi_label_classification",
).to(device)
wav2vec2_emotion_model.eval()


def compute_final_emotion(wav2vec2_probs, kobert_probs, kobert_labels):
    # 음성 기반에서 가장 강한 감정 선택
    wav_top_index = int(torch.tensor(wav2vec2_probs).argmax())
    wav_top_emotion = wav2vec2_labels[wav_top_index]

    if wav_top_emotion not in EMOTION_EN_TO_INDEX:
        return "Unknown"

    wav_idx = EMOTION_EN_TO_INDEX[wav_top_emotion]

    final_scores = []
    for i, label in enumerate(kobert_labels):
        if label == "Normal":  # 제외
            final_scores.append(0.0)
            continue
        if label == "Aversion":
            label = "Disgust"  # 혼합 허용도 매핑 위해 이름 통일

        if label not in EMOTION_EN_TO_INDEX:
            final_scores.append(0.0)
            continue

        ko_idx = EMOTION_EN_TO_INDEX[label]
        compatibility = MIX_MATRIX[ko_idx][wav_idx]
        weighted_score = (kobert_probs[i] + compatibility) / 2
        final_scores.append(weighted_score)

    return torch.tensor(final_scores).tolist()


def full_multimodal_analysis(audio_path: str):
    # soundfile.read → mono → 16kHz 리샘플링
    audio_np, sr = sf.read(audio_path)
    if audio_np.ndim > 1:
        audio_np = np.mean(audio_np, axis=1)
    if sr != 16000:
        audio_np = librosa.resample(audio_np, orig_sr=sr, target_sr=16000)
        sr = 16000

    # Whisper → 텍스트 생성
    whisper_inputs = processor(audio_np, sampling_rate=16000, return_tensors="pt").to(
        device
    )
    with torch.no_grad():
        generated_ids = whisper_model.generate(
            **whisper_inputs, max_new_tokens=256, num_beams=5, no_repeat_ngram_size=3
        )
    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=128,
    ).to(kmodel.device)

    with torch.no_grad():
        outputs = kmodel(**inputs)
        logits = outputs.logits
        probs = torch.sigmoid(logits).cpu().numpy()[0]

    emotion_text = probs.tolist()
    # api 텍스트 요약
    summary = summarize(text)

    # Wav2Vec2 기반 음성 감정 분류
    w2v_inputs = wav2vec2_processor(
        audio_np,
        sampling_rate=16000,
        return_tensors="pt",
        padding=True,
    ).to(device)

    with torch.no_grad():
        w2v_outputs = wav2vec2_emotion_model(**w2v_inputs)
        w2v_probs_arr = w2v_outputs.logits.squeeze().cpu().numpy()

    emotion_audio = [float(w2v_probs_arr[i]) for i in range(len(wav2vec2_labels))]

    final_emotion = compute_final_emotion(emotion_audio, emotion_text, kobert_labels)

    return summary, final_emotion
