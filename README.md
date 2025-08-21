Emotion Analysis Service — README

음성(오디오)과 텍스트를 함께 사용해 감정을 분석하고, 공감형 요약을 생성하는 파이프라인입니다.
utils.py 하나로 동작하며, 모델/가중치/베이스라인 벡터는 Hugging Face Hub에서 자동으로 다운로드합니다.

구성 요소

STT: faster-whisper (large-v3)

텍스트 감정 분류: HyukII/text-emotion-model (Transformers)

오디오 감정 분류: HyukII/audio-emotion-model

PyTorch 커스텀 모델(model.py) + 가중치(pytorch_model.pth)

베이스라인 벡터(baseline_mean_*.npy, baseline_std_*.npy) — (X−mean)/(std+1e−8)

요약: OpenAI Chat Completions (gpt-4o-mini) – 선택적

설치
# Python 3.9+
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu  # (GPU면 본인 환경에 맞게)
pip install transformers huggingface_hub safetensors
pip install faster-whisper librosa soundfile numpy
pip install openai  # 요약을 쓸 경우


개발 서버에서 토크나이저 경고를 줄이려면 환경변수 설정:
TOKENIZERS_PARALLELISM=false

환경 변수

OPENAI_API_KEY (선택): 요약 생성에 필요. 없으면 "summary": "(요약 실패) ..." 형태로 반환됨.

현재 utils.py는 아래 고정된 HF 리포를 사용합니다.

텍스트: HyukII/text-emotion-model

오디오 : HyukII/audio-emotion-model

입력/출력 형식
입력

음성 파일: .wav (다른 샘플링이어도 OK)

내부에서 ensure_16k_mono()로 16kHz mono 변환 후 처리

옵션 파라미터:

gender: "MALE" 또는 "FEMALE" (오디오 정규화용 베이스라인 선택)

lang: STT 언어 코드 (기본 "ko")

출력(JSON)
{
  "summary": "공감형 한두 문장 요약(또는 요약 실패 메시지)",
  "emotion_analysis": [p0, p1, p2, p3, p4, p5]
}


emotion_analysis는 한국어 6라벨 순서로 확률(0~1) 배열을 제공합니다.
라벨 순서는 텍스트 모델의 id2label에서 가져오며, 기본은:

['기쁨','당황','분노','불안','상처','슬픔']

동작 흐름

STT: Whisper로 업로드된 음성을 텍스트로 전사

텍스트 감정: 문장 단위 추론 → 감정 비율(%) 산출(KO 6라벨)

오디오 감정:

MFCC(13ch) 추출, 길이 100 프레임으로 패딩/자르기 → (100,13)

성별별 베이스라인(mean/std)을 HF에서 .npy로 다운로드 후 delta=(X-mean)/(std+1e-8)

텐서 (1,13,100)로 변환하여 오디오 모델 추론 → EN 6라벨 확률

라벨 매핑 & 융합:

KO(텍스트) → EN(오디오)로 투영 후, 텍스트 0.7 + 오디오 0.3 가중합

다시 EN → KO로 근사 역투영 → 최종 KO 확률 벡터

요약: 입력 텍스트를 공감형으로 1–2문장 요약(옵션)

빠른 사용 예시 (Django view)
# views.py
from django.http import JsonResponse
from .utils import run_pipeline_on_uploaded_file

def analyze_view(request):
    if request.method != "POST" or "file" not in request.FILES:
        return JsonResponse({"error": "POST 파일 업로드 필요"}, status=400)

    gender = request.POST.get("gender", "MALE")  # "MALE" | "FEMALE"
    lang = request.POST.get("lang", "ko")
    result = run_pipeline_on_uploaded_file(request.FILES["file"], gender=gender, lang=lang)
    return JsonResponse(result, json_dumps_params={"ensure_ascii": False})


curl 테스트:

curl -X POST http://localhost:8000/api/diary/analyze \
  -F "file=@/path/to/sample.wav" \
  -F "gender=MALE" \
  -F "lang=ko"

스탠드얼론 테스트 (파이프라인 함수 직접 호출)
from pathlib import Path
from types import SimpleNamespace
from .utils import run_pipeline_on_uploaded_file

class FileObj:
    def __init__(self, p): self.p=p
    def chunks(self, size=8192):
        with open(self.p, "rb") as f:
            while True:
                b=f.read(size)
                if not b: break
                yield b

wav = FileObj("/path/to/sample.wav")
res = run_pipeline_on_uploaded_file(wav, gender="FEMALE", lang="ko")
print(res)

내부 구현 상세 (핵심 포인트)

텍스트 모델 로딩

_tok = AutoTokenizer.from_pretrained("HyukII/text-emotion-model", use_fast=True)
_text_model = AutoModelForSequenceClassification.from_pretrained(...).eval()
# id2label은 config.json 우선, 없으면 label_map.json 다운로드 사용


오디오 모델/라벨/베이스라인 로딩 (HF 직행)

from huggingface_hub import hf_hub_download
from importlib.machinery import SourceFileLoader

# labels.json 다운로드 → 리스트
# model.py 다운로드 → SourceFileLoader로 PyTorchAudioModel 로드
# pytorch_model.pth 다운로드 → state_dict 로드
# baseline_mean|std_{male|female}.npy 다운로드 → delta 정규화


오디오 입력 & 정규화

MFCC: librosa.feature.mfcc(..., n_mfcc=13).T → (T,13)

길이 100 프레임으로 패딩/슬라이스 → (100,13)

delta = (X - mean) / (std + 1e-8)

텐서 (1,13,100)로 변환 후 모델 추론

라벨 매핑 행렬

EN 라벨(오디오): labels.json (기본 ["ANGRY","SAD","DISGUST","HAPPY","FEAR","SURPRISE"])

KO 라벨(텍스트): ['기쁨','당황','분노','불안','상처','슬픔']

텍스트 0.7, 오디오 0.3 비율로 융합 (원하면 fuse_text_audio(..., w_text, w_audio) 조정)

주의/팁

요약 비활성화: OPENAI_API_KEY가 없으면 자동으로 실패 메시지를 넣고 넘어갑니다.
운영에서 요약이 꼭 필요 없다면 empathetic_summary()를 건너뛰도록 수정하세요.

GPU 사용: torch.cuda.is_available()에 따라 자동 선택. Whisper compute_type도 자동 조정(float16/int8).

캐시: huggingface_hub는 다운로드 파일을 로컬 캐시에 보관합니다(오프라인 재사용 가능).

경고 억제: TOKENIZERS_PARALLELISM=false 권장(Django dev server의 autoreload/fork 시 경고 방지).

라벨 순서 불일치 주의:

오디오 labels.json과 매핑 행렬(EN)의 순서가 반드시 일치해야 합니다.

텍스트 라벨 순서는 텍스트 모델의 id2label에 따릅니다.

라이선스 & 크레딧

모델과 코드의 라이선스는 각 HF 리포의 README.md/LICENSE를 따릅니다.

Whisper 모델: faster-whisper

Transformers & HF Hub: 🤗 Hugging Face
