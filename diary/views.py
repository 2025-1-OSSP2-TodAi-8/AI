from django.http import JsonResponse
from django.views.decorators.http import require_POST
from django.views.decorators.csrf import csrf_exempt
from .utils import run_pipeline_on_uploaded_file


@csrf_exempt
@require_POST
def analyze(request):
    """
    form-data:
      - audio: 업로드 음성 파일
      - gender: MALE | FEMALE (옵션, 기본 MALE)
    """
    audio = request.FILES.get("audio")
    if not audio:
        return JsonResponse({"error": "audio 파일(form-data) 누락"}, status=400)

    gender = (request.POST.get("gender") or "MALE").upper()
    if gender not in ("MALE", "FEMALE"):
        gender = "MALE"

    result = run_pipeline_on_uploaded_file(audio, gender=gender, lang="ko")
    return JsonResponse(result, json_dumps_params={"ensure_ascii": False})
