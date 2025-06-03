# emotions/utils.py

import os
import torch
import numpy as np
import soundfile as sf
import librosa

from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForSeq2SeqLM,
    Wav2Vec2Processor,
    Wav2Vec2ForSequenceClassification,
)

# 장치 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Whisper 모델 로드
WHISPER_DIR = os.path.join(os.path.dirname(__file__), "whisper-base")
processor = WhisperProcessor.from_pretrained(WHISPER_DIR, local_files_only=True)
whisper_model = WhisperForConditionalGeneration.from_pretrained(
    WHISPER_DIR, local_files_only=True
).to(device)
whisper_model.eval()

# KoBERT 감정 분류 모델 로드
KOBERT_EMOTION_DIR = os.path.join(os.path.dirname(__file__), "ke")
emotion_tokenizer = AutoTokenizer.from_pretrained(
    KOBERT_EMOTION_DIR, local_files_only=True
)
emotion_model = AutoModelForSequenceClassification.from_pretrained(
    KOBERT_EMOTION_DIR, local_files_only=True
).to(device)
emotion_model.eval()

original_labels = ["기쁨", "불안", "당황", "분노", "슬픔", "상처"]
mapped_labels = {
    "기쁨": "기쁨",
    "슬픔": "슬픔",
    "분노": "분노",
    "당황": "놀람",
    "불안": "공포",
    "상처": "혐오",
}

wav2vec2_labels = ["Angry", "Sadness", "Disgust", "Happiness", "Fear", "Surprise"]

# KoBART 요약 모델 로드
KOBART_SUMMARY_DIR = os.path.join(os.path.dirname(__file__), "ks")
summary_tokenizer = AutoTokenizer.from_pretrained(
    KOBART_SUMMARY_DIR, local_files_only=True
)
summary_model = AutoModelForSeq2SeqLM.from_pretrained(
    KOBART_SUMMARY_DIR, local_files_only=True
).to(device)
summary_model.eval()

# Wav2Vec2 기반 음성 감정 분류기 로드
SAVED_W2VEMO_DIR = os.path.join(os.path.dirname(__file__), "saved_model2")
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
    emo_inputs = emotion_tokenizer(
        text, return_tensors="pt", truncation=True, padding="max_length", max_length=512
    ).to(device)

    with torch.no_grad():
        emo_logits = emotion_model(**emo_inputs).logits
    emo_probs_arr = torch.softmax(emo_logits, dim=-1).squeeze().cpu().numpy()

    original_probs = {
        original_labels[i]: float(emo_probs_arr[i]) for i in range(len(original_labels))
    }
    emotion_probs = {mapped_labels[k]: v for k, v in original_probs.items()}

    # KoBART 텍스트 요약
    sum_inputs = summary_tokenizer(
        text, return_tensors="pt", truncation=True, padding="longest", max_length=512
    )
    if "token_type_ids" in sum_inputs:
        sum_inputs.pop("token_type_ids")
    sum_inputs = {k: v.to(device) for k, v in sum_inputs.items()}

    with torch.no_grad():
        summary_ids = summary_model.generate(
            **sum_inputs,
            max_length=100,
            min_length=20,
            num_beams=4,
            early_stopping=True,
        )
    summary = summary_tokenizer.decode(summary_ids[0], skip_special_tokens=True)

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

    emotion_prob2 = [float(w2v_probs_arr[i]) for i in range(len(wav2vec2_labels))]

    return text, summary, emotion_prob2
