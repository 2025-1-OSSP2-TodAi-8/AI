# emotions/utils.py

import os
import torch
import numpy as np
import soundfile as sf
import librosa
import openai

from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Wav2Vec2Processor,
    Wav2Vec2ForSequenceClassification,
    BertConfig,
    BertTokenizer,
    BertForSequenceClassification,
)


emotion_labels = [
    "Fear",
    "Surprise",
    "Angry",
    "Sadness",
    "Normal",
    "Happiness",
    "Aversion",
]
wav2vec2_labels = ["Angry", "Sadness", "Disgust", "Happiness", "Fear", "Surprise"]

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
# 2) emotion-text 디렉터리
EMO_DIR = os.path.join(os.path.dirname(__file__), "emotion-text")

# 3) 토크나이저: vocab.txt 한 파일만으로 로드
tokenizer = BertTokenizer.from_pretrained(
    os.path.join(EMO_DIR, "vocab.txt"), local_files_only=True
)

config = BertConfig.from_json_file(os.path.join(EMO_DIR, "config.json"))
config.num_labels = 7
model = BertForSequenceClassification(config)
state_dict = torch.load(os.path.join(EMO_DIR, "model.pt"), map_location=device)
model.load_state_dict(state_dict)

model.eval()

# Wav2Vec2 기반 음성 감정 분류기 로드
SAVED_W2VEMO_DIR = os.path.join(os.path.dirname(__file__), "emotion-audio")
wav2vec2_processor = Wav2Vec2Processor.from_pretrained(
    SAVED_W2VEMO_DIR, local_files_only=True
)
wav2vec2_emotion_model = Wav2Vec2ForSequenceClassification.from_pretrained(
    SAVED_W2VEMO_DIR, local_files_only=True
).to(device)
wav2vec2_emotion_model.eval()


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

    # KoBERT 텍스트 감정 분류
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=512,
    ).to(device)

    with torch.no_grad():
        logits = model(**inputs).logits

    probs = torch.softmax(logits, dim=-1).squeeze().cpu().numpy()
    emotion_text = {
        emotion_labels[i]: float(probs[i]) for i in range(len(emotion_labels))
    }

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

    return text, summary, emotion_text, emotion_audio
