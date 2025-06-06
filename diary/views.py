import datetime
from rest_framework.decorators import api_view, permission_classes, parser_classes
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework import status

from .models import Diary

from .utils import full_multimodal_analysis, wav2vec2_labels


@api_view(["POST"])
@permission_classes([IsAuthenticated])
@parser_classes([MultiPartParser, FormParser])
def record(request):
    user = request.user
    date_str = request.data.get("date")
    audio = request.FILES.get("audio")

    if not date_str or not audio:
        return Response(
            {"success": 0, "emotion": [], "text": "데이터 누락"},
            status=status.HTTP_400_BAD_REQUEST,
        )

    try:
        date = datetime.datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError:
        return Response(
            {"success": 0, "emotion": [], "text": "날짜형식 불일치"},
            status=status.HTTP_400_BAD_REQUEST,
        )

    diary = Diary.objects.filter(user=user, date=date).first()

    if diary:
        diary.audio = audio
        diary.date = date
        diary.save(update_fields=["audio", "date"])
    else:
        diary = Diary.objects.create(
            user=user, date=date, audio=audio, emotion=[0, 0, 0, 0, 0, 0, 0]
        )

    try:
        text, summary, emotion_prob1, emotion_prob2 = full_multimodal_analysis(
            diary.audio.path
        )
    except Exception:
        return Response(
            {"success": 0, "emotion": diary.emotion, "text": "파이프라인 에러"},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )
    diary.summary = summary
    diary.emotion = emotion_prob2
    diary.save()

    return Response(
        {
            "success": 1,
            "koBERT": emotion_prob1,
            "wav2vec2 감정 순서": wav2vec2_labels,
            "wav2vec2": emotion_prob2,
            "text": diary.summary or "",
        },
        status=status.HTTP_200_OK,
    )


@api_view(["POST"])
@permission_classes([IsAuthenticated])
def marking(request):

    user = request.user
    date_str = request.data.get("date")
    if not date_str:
        return Response(
            {"success": 0, "message": "date 누락"},
            status=status.HTTP_400_BAD_REQUEST,
        )

    # 날짜 파싱
    try:
        date = datetime.datetime.strptime(date_str, "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return Response(
            {"success": 0, "message": "잘못된 date 형식 (YYYY-MM-DD)"},
            status=status.HTTP_400_BAD_REQUEST,
        )

    diary = Diary.objects.filter(user=user, date=date).first()

    if not diary:
        return Response(
            {"success": 0, "message": f"{date_str}의 일기를 찾을 수 없습니다."},
            status=status.HTTP_404_NOT_FOUND,
        )

    diary.marking = not diary.marking
    diary.save(update_fields=["marking"])

    return Response(
        {"success": 1, "marking": diary.marking},
        status=status.HTTP_200_OK,
    )


EMOTION_LABELS = ["행복", "슬픔", "놀람", "화남", "혐오", "공포", "중립"]


@api_view(["POST"])
@permission_classes([IsAuthenticated])
def get_marked_year(request):
    user = request.user
    year = request.data.get("year")

    if year is None:
        return Response({"error": "year 필수가 누락되었습니다."}, status=400)
    try:
        year = int(year)
    except (ValueError, TypeError):
        return Response({"error": "year는 정수여야 합니다."}, status=400)

    diaries = Diary.objects.filter(
        user=user,
        marking=True,
        date__year=year,
    ).order_by("date")

    emotions = []
    for d in diaries:
        probs = d.emotion or []
        idx = max(range(len(probs)), key=lambda i: probs[i])
        label = EMOTION_LABELS[idx]

        emotions.append({"date": d.date.isoformat(), "emotion": label})

    return Response({"emotions": emotions}, status=200)


@api_view(["POST"])
@permission_classes([IsAuthenticated])
def get_marked_month(request):
    user = request.user
    year = request.data.get("year")
    month = request.data.get("month")

    if year is None or month is None:
        return Response({"error": "year or month 필수가 누락되었습니다."}, status=400)
    try:
        year = int(year)
        month = int(month)
    except (ValueError, TypeError):
        return Response({"error": "year, month는 정수여야 합니다."}, status=400)

    diaries = Diary.objects.filter(
        user=user,
        marking=True,
        date__year=year,
        date__month=month,
    ).order_by("date")

    emotions = []
    for d in diaries:
        probs = d.emotion or []
        if len(probs) == len(EMOTION_LABELS):
            idx = max(range(len(probs)), key=lambda i: probs[i])
            label = EMOTION_LABELS[idx]
        else:
            label = ""
        emotions.append({"date": d.date.isoformat(), "emotion": label})

    return Response({"emotions": emotions}, status=200)
