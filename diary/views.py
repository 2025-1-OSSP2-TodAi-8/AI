import tempfile
import os
from rest_framework.decorators import api_view, parser_classes
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.response import Response
from rest_framework import status
from .utils import full_multimodal_analysis


@api_view(["POST"])
@parser_classes([MultiPartParser, FormParser])
def analyze(request):
    """
    음성 파일(wav)을 업로드 받아 텍스트 요약 및 감정 분석 결과를 반환합니다.
    """
    audio_file = request.FILES.get("audio")

    if not audio_file:
        return Response(
            {"error": "음성 파일이 제출되지 않았습니다."},
            status=status.HTTP_400_BAD_REQUEST,
        )

    temp_wav_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_wav:
            for chunk in audio_file.chunks():
                temp_wav.write(chunk)
            temp_wav_path = temp_wav.name
        summary, final_emotion = full_multimodal_analysis(temp_wav_path)

        response_data = {"summary": summary, "emotion": final_emotion}

        return Response(response_data, status=status.HTTP_200_OK)

    except Exception as e:
        print(f"Error during audio analysis: {e}")
        return Response(
            {"error": "음성 파일을 처리하는 중 오류가 발생했습니다."},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )

    finally:
        if temp_wav_path and os.path.exists(temp_wav_path):
            os.remove(temp_wav_path)
