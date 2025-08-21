# 감정 분석 AI엔진 

📌 프로젝트 개요

본 프로젝트는 한국어 일기 텍스트와 음성 데이터를 활용한 감정 분석 앱을 개발하는 과정에서 사용된 텍스트 기반 감정 분류 모델을 포함하고 있다. 
이는 앱 시나리오의 핵심 기능인 “오늘 하루는 어떠셨나요?” 질문에 대한 감정 분석 단계에 해당한다.

___________________________________________________________________________

## 🎤 1. 음성 분석 모델 

### 📌 1. 개요
- 사용자의 녹음 음성을 입력받아 음향적 특징을 추출하고 감정을 분류하는 모델
- 텍스트 분석 모델과 달리, 목소리의 억양·속도·에너지·스펙트럼 변화 등을 활용해 감정을 감지
- 일기 텍스트가 긍정적으로 작성되더라도, 목소리 톤이 우울하다면 실제 감정을 보완적으로 파악할 수 있음
  


### 🛠️ 2. 특징 추출 (Feature Extraction)
본 프로젝트에서는 음성에서 MFCC(Mel-Frequency Cepstral Coefficients)정보를 추출하고 있음
- MFCC (Mel-Frequency Cepstral Coefficients): 음성의 주파수 스펙트럼을 요약한 13차원 계수
- 고정된 시퀀스 길이: 100 프레임으로 맞추어 CNN-LSTM 모델에 입력 가능
- 패딩 / 자르기 처리: 발화 길이가 짧으면 0으로 패딩, 길면 잘라냄
- 사용 라이브러리: librosa, numpy



### 🔎 3. 모델 구조
- CNN + BiLSTM 기반 시퀀스 모델
- Conv1D → 음향 스펙트럼 특징 추출
- BiLSTM → 시간적 변화 패턴 학습
- Dense + Softmax → 감정 클래스 확률 출력


### ⚙️ 4.학습 방법

4.1. 데이터셋 구성
- 훈련 데이터 : AiHub의 '감정 음성 데이터셋' (https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=637)
- 사용자 음성 → 9~13차원 특징 벡터(MFCC 등) 시퀀스로 변환
- 레이블: 감정 클래스 (예: JOY, SAD, ANGRY 등)
- 불균형 보완: 데이터 증강(속도 변환, 피치 쉬프트)

4.2. 훈련 전략
- 입력: (시퀀스 길이, feature_dim) 형태의 음성 특징 시퀀스
- 출력: 감정 확률 분포 (Softmax)
- 손실 함수: CrossEntropyLoss
- 옵티마이저: AdamW, learning rate scheduling 적용

4.3. 중립 벡터 기반 분석 (Delta Approach)
- 남자, 여자용  Neutral Baseline Vector를 먼저 저장
- 새로운 발화 입력 시 → Δ = (현재 벡터 – baseline) 계산
- Δ 벡터를 모델에 입력하여 개인화된 감정 예측 가능


### 🚀 5. 앱에서의 활용
- 사용자가 앱 실행 시 음성을 녹음 → 실시간 특징 추출
- 감정 모델이 분류한 결과를 텍스트 기반 감정과 비교
- 두 결과가 불일치할 경우: “텍스트는 기쁨이지만 목소리가 우울해 보이네요. 괜찮으신가요?”	“분노가 감지되었지만 목소리가 차분하시네요. 혹시 화가 누그러드셨나요?”
- 보호자에게는 장기적 음성 패턴 변화를 기반으로 경고 알림 전송 ￼






__________________________________________________________________________________________


## 2. 📝 텍스트 기반 감정 분석 모델

### 📌 1. 개요
- 사용자가 작성하거나 음성에서 변환된 텍스트 데이터를 입력으로 받아 감정을 분류하는 모델.
- 한국어에 특화된 klue/roberta-base 사전학습 언어모델을 파인튜닝하여 구현되었다.
- 앱에서 녹음된 음성은 STT(Speech-to-Text) 과정을 거쳐 텍스트로 변환되며, 택스트 모델의 입력값으로 사용된다.



### ⚙️ 2. 모델 구조
- Pre-trained Model: KLUE RoBERTa Base
- Fine-tuning Task: Multi-class text classification
- 출력 라벨 예시:	['ANGRY', 'SAD', 'JOY', 'FEAR', 'SURPRISE', 'DISGUST']



### 🛠️ 3. 학습 방법
1.	데이터 준비
- AiHub의 '공감형 대화' (https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=7130)
- 라벨을 숫자로 매핑 (label2id, id2label)

2.	토큰화

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("klue/roberta-base")

def preprocess(ex):
    return tokenizer(
        ex["text"], 
        padding="max_length", 
        truncation=True, 
        max_length=256
    )

- 최대 길이: 256
- Padding/Truncation 적용

3.모델 불러오기

from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(
    "klue/roberta-base",
    num_labels=len(labels),
    id2label=id2label,
    label2id=label2id
)



📊 성능 평가
	•	학습 중 매 epoch마다 Accuracy와 Macro-F1을 측정
	•	최종적으로 F1-macro 기준으로 성능이 가장 좋은 모델을 저장 및 로드
	•	클래스 불균형 데이터에서도 안정적인 성능 확보


🚀 활용 방안
	•	STT로 변환된 일기 텍스트 → 본 모델로 감정 분석
	•	텍스트 기반 결과 + 음성 기반 결과를 종합하여 최종 감정 판단
	•	사용자가 긍정적으로 기록했더라도, 목소리에서 우울감이 감지될 경우 보완적 해석 가능

'
