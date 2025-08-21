# 감정 분석 파이프라인

이 프로젝트는 **음성 파일을 입력 받아 텍스트(STT 변환)와 오디오 특징**을 기반으로 감정을 분석하고, 결과를  
**[HAPPY, SAD, ANGRY, SURPRISE, FEAR, DISGUST]** 순서의 확률 벡터로 반환하는 파이프라인입니다.  
또한 OpenAI GPT 모델을 활용하여 입력된 음성을 공감적인 요약 문장으로 생성합니다.  

---

## 처리 흐름

1. **업로드된 음성 파일 저장**
   - Django `UploadedFile` 객체를 임시 디렉토리에 저장

2. **16kHz Mono 변환**
   - Whisper 입력 포맷에 맞게 변환 (`librosa`, `soundfile` 활용)

3. **음성 인식 (STT)**
   - [faster-whisper](https://github.com/guillaumekln/faster-whisper)로 한국어 음성 → 텍스트 변환

4. **텍스트 감정 분석**
   - `HyukII/text-emotion-model` (model repository 참고)
   - 감정 레이블: `['기쁨','당황','분노','불안','상처','슬픔']`

5. **오디오 감정 분석**
   - `HyukII/audio-emotion-model` (model repository 참고)
   - 모델 출력 레이블: `[ANGRY, SAD, DISGUST, HAPPY, FEAR, SURPRISE]`

6. **라벨 매핑 & 융합**
   - 텍스트(KO) 결과를 EN 감정 공간으로 투영
   - 오디오(EN) 결과와 가중 평균 (텍스트 70%, 오디오 30%)

7. **최종 EN 순서로 정렬**
   - `[ANGRY, SAD, DISGUST, HAPPY, FEAR, SURPRISE]` →  
     `[HAPPY, SAD, ANGRY, SURPRISE, FEAR, DISGUST]`      
     해당변환은 학습 시킨 모델의 출력 양식에 따라 적절히 수정하여 사용될 수 있습니다.

8. **공감 요약 생성**
   - OpenAI GPT(`gpt-4o-mini`)로 공감적인 한국어 요약 생성

---
## 요청 형식 (Input Format)

입력: Django UploadedFile 객체 (예: request.FILES['audio'])

지원 파일 형식: .wav

권장 샘플링 레이트: 16kHz mono (다른 경우 자동 변환됨)

``` json
{
	"gender":"MALE or FEMALE",
	"audio":음성파일.wav
}
```

## 반환 형식

```json
{
  "summary": "햇살 좋은 날에 뛰어다니며 가벼운 기분을 느끼고 웃음이 절로 나는 행복한 하루를 보냈군요. 이 기분을 오랫동안 간직하고 싶어 하셨을 것 같아요.",
  "emotion": [0.52, 0.12, 0.08, 0.15, 0.10, 0.03],
  "emotion_labels": ["HAPPY", "SAD", "ANGRY", "SURPRISE", "FEAR", "DISGUST"]
}
```
summary: 공감적인 한국어 요약 문장

emotion: [HAPPY, SAD, ANGRY, SURPRISE, FEAR, DISGUST] 순서의 확률 값

emotion_labels: 확률 벡터와 매칭되는 레이블 순서

## 기술 스택
STT: faster-whisper

텍스트 감정 분석: HuggingFace Transformers (HyukII/text-emotion-model)

오디오 감정 분석: PyTorch + HuggingFace Hub (HyukII/audio-emotion-model)

특징 추출: librosa (MFCC)

요약: OpenAI GPT (gpt-4o-mini)

## 실행 방법
1. 환경 변수 설정
``` bash
export OPENAI_API_KEY=your_openai_api_key
```
2. Python 의존성 설치
``` bash
pip install -r requirements.txt
```
3. 파이프라인 실행
``` python
from pipeline import run_pipeline_on_uploaded_file

result = run_pipeline_on_uploaded_file(django_file, gender="MALE", lang="ko")
print(result)
```

## 라이선스
이 프로젝트는 오픈소스로 공개됩니다.  자유롭게 사용 및 기여해 주세요.
---
