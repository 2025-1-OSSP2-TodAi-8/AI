from rest_framework.decorators import api_view
from django.http import JsonResponse
from .utils import run_pipeline_on_uploaded_file


@api_view(["POST"])
def analyze(request):
    """
    POST /analyze-emotion
    form-data: audio=<파일>
    """
    audio = request.FILES.get("audio")
    if not audio:
        return JsonResponse({"error": "audio 파일이 필요합니다"}, status=400)

    result = run_pipeline_on_uploaded_file(audio, lang="ko")
    return JsonResponse(result, json_dumps_params={"ensure_ascii": False})
