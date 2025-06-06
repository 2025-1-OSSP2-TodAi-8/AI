import datetime
from django.views.decorators.http import require_POST
from django.views.decorators.csrf import csrf_exempt
from rest_framework.response import Response
from diary.models import Diary
from .models import Emotion
from people.models import People

from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated


EMOTION_LABELS = ["행복", "슬픔", "놀람", "화남", "혐오", "공포", "중립"]


@api_view(["POST"])
@permission_classes([IsAuthenticated])
def emotions_day(request):
    user = request.user

    year = request.data.get("year")
    month = request.data.get("month")
    day = request.data.get("day")

    if year is None or month is None or day is None:
        return (
            Response(
                {"error": "year, month, day 모두 필요합니다."},
                status=400,
            ),
        )

    try:
        target_date = datetime.date(int(year), int(month), int(day))
    except (ValueError, TypeError):
        return Response(
            {"error": "유효한 날짜(year, month, day) 형식이 아닙니다."},
            status=400,
        )

    try:
        diary = Diary.objects.get(user=user, date=target_date)
    except Diary.DoesNotExist:
        return Response(
            {"error": f"{target_date.isoformat()}에 저장된 일기가 없습니다."},
            status=404,
        )

    probs = diary.emotion or []
    
    max_idx = max(range(len(probs)), key=lambda i: probs[i])
    label = EMOTION_LABELS[max_idx]

    return Response(
        {
            "date": target_date.isoformat(),
            "emotion": label,
            "emotion_rate": diary.emotion,
            "summary": diary.summary,
        },
        status=200,
    )


@api_view(["POST"])
@permission_classes([IsAuthenticated])
def emotions_month(request):
    user = request.user

    year = request.data.get("year")
    month = request.data.get("month")

    if year is None or month is None:
        return Response({"error": "year, month 모두 필요합니다."}, status=400)

    try:
        year = int(year)
        month = int(month)
        # 유효한 달(month)인지 확인
        if not 1 <= month <= 12:
            raise ValueError
    except (ValueError, TypeError):
        return Response(
            {"error": "year, month는 정수이며 month는 1~12여야 합니다."}, status=400
        )

    diaries = Diary.objects.filter(
        user=user, date__year=year, date__month=month
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


@api_view(["POST"])
@permission_classes([IsAuthenticated])
def emotions_year(request):
    user = request.user

    year = request.data.get("year")

    if year is None:
        return Response({"error": "year, month 모두 필요합니다."}, status=400)

    try:
        year = int(year)
    except TypeError:
        return Response({"error": "year는 정수이어야 합니다."}, status=400)

    diaries = Diary.objects.filter(user=user, date__year=year).order_by("date")

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
