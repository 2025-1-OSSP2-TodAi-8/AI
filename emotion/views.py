import json
from django.shortcuts import render
from django.views.decorators.http import require_POST
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse, HttpResponseBadRequest, HttpResponseNotFound
from .models import Emotion
from people.models import People


def _get_user(user_id):
    try:
        return People.objects.get(pk=int(user_id))
    except (People.DoesNotExist, ValueError):
        return None

@csrf_exempt
@require_POST
def emotions_day(request):
    """
    POST /api/emotion/day/
    JSON: { user_id: int, year: int, month: int, day: int }
    Response: { date: "YYYY-MM-DD", emotion: "감정" }
    """

    try:
        payload = json.loads(request.body)
        user_id = payload["user_id"]
        y = int(payload["year"])
        m = int(payload["month"])
        d = int(payload["day"])
    except (KeyError, ValueError, json.JSONDecodeError):
        return HttpResponseBadRequest("json 구조 불일치")

    user = _get_user(user_id)
    if not user:
        return HttpResponseBadRequest("존재하지 않는 사용자")

    try:
        emotion = Emotion.objects.get(
            user=user, date__year=y, date__month=m, date__day=d
        )
    except Emotion.DoesNotExist:
        return HttpResponseNotFound("해당 날짜의 감정 기록이 없음")

    return JsonResponse(
        {"date": emotion.date.strftime("%Y-%m-%d"), "emotion": emotion.emotion}
    )

@csrf_exempt
@require_POST
def emotions_month(request):
    """
    POST /api/emotion/month/
    JSON: { user_id: int, year: int, month: int }
    Response: { emotions: [ { date: "YYYY-MM-DD", emotion: "감정" }, … ] }
    """
    try:
        payload = json.loads(request.body)
        user_id = payload["user_id"]
        y = int(payload["year"])
        m = int(payload["month"])
    except (KeyError, ValueError, json.JSONDecodeError):
        return HttpResponseBadRequest("json 구조 불일치")

    user = _get_user(user_id)
    if not user:
        return HttpResponseBadRequest("존재하지 않는 사용자")

    qs = Emotion.objects.filter(user=user, date__year=y, date__month=m).order_by("date")

    emotions = list(qs.values("date", "emotion"))
    return JsonResponse({"emotions": emotions})

@csrf_exempt
@require_POST
def emotions_year(request):
    """
    POST /api/emotions/year/
    JSON: { user_id: int, year: int }
    Response: { emotions: [ { date: "YYYY-MM-DD", emotion: "감정" }, … ] }
    """
    try:
        payload = json.loads(request.body)
        user_id = payload["user_id"]
        y = int(payload["year"])
    except (KeyError, ValueError, json.JSONDecodeError):
        return HttpResponseBadRequest("json 구조 불일치")

    user = _get_user(user_id)
    if not user:
        return HttpResponseBadRequest("존재하지 않는 사용자")

    qs = Emotion.objects.filter(user=user, date__year=y).order_by("date")

    emotions = list(qs.values("date", "emotion"))
    return JsonResponse({"emotions": emotions})
