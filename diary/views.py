from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status, parsers, permissions
from people.models import People
from emotion.models import Emotion
from .models import Diary
import datetime


class RecordAPIView(APIView):
    permission_classes = [permissions.AllowAny]
    parser_classes = [parsers.MultiPartParser, parsers.FormParser]

    def post(self, request, *args, **kwargs):
        user_id = request.data.get("user_id")
        audio = request.FILES.get("audio")
        date_str = request.data.get("date")

        if not all([user_id, audio, date_str]):
            return Response(
                {"success": 0, "emotion": [], "text": "항목 누락"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        try:
            user = People.objects.get(pk=user_id)
        except People.DoesNotExist:
            return Response(
                {"success": 0, "emotion": [], "text": "존재하지 않는 유저"},
                status=status.HTTP_404_NOT_FOUND,
            )

        try:
            diary_date = datetime.datetime.strptime(date_str, "%Y-%m-%d").date()
        except ValueError:
            return Response(
                {"success": 0, "emotion": [], "text": "잘못된 날짜 형식 (YYYY-MM-DD)"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        emotion_obj, created = Emotion.objects.get_or_create(
            user=user, date=diary_date, defaults={"emotion": ""}
        )

        diary = Diary.objects.create(
            user=user, audio=audio, date=diary_date, emotion=emotion_obj
        )

        return Response(
            {"success": 1, "emotion": [0, 0, 0, 0, 0, 0, 0], "text": ""},
            status=status.HTTP_200_OK,
        )


class MarkingAPIView(APIView):
    permission_classes = [permissions.AllowAny]
    parser_classes = [parsers.JSONParser, parsers.FormParser]

    def post(self, request, *args, **kwargs):
        user_id = request.data.get("user_id")
        date_str = request.data.get("date")
        People.objects.get(pk=user_id)

        if not all([user_id, date_str]):
            return Response(
                {"success": 0, "message": "필드 누락"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        try:
            user = People.objects.get(pk=user_id)
        except People.DoesNotExist:
            return Response(
                {"success": 0, "message": "존재하지 않는 사용자"},
                status=status.HTTP_404_NOT_FOUND,
            )

        try:
            target_date = datetime.datetime.strptime(date_str, "%Y-%m-%d").date()
        except ValueError:
            return Response(
                {"success": 0, "message": "date 형식 오류"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        try:
            diary = Diary.objects.get(user=user, date=target_date)
        except Diary.DoesNotExist:
            return Response(
                {"success": 0, "message": "해당 날짜의 일기를 찾을 수 없음"},
                status=status.HTTP_404_NOT_FOUND,
            )

        diary.marking = not diary.marking
        diary.save(update_fields=["marking"])

        return Response(
            {
                "success": 1,
            },
            status=status.HTTP_200_OK,
        )


class MarkedYearAPIView(APIView):
    permission_classes = [permissions.AllowAny]
    parser_classes = [parsers.JSONParser, parsers.FormParser]

    def post(self, request, *args, **kwargs):
        user_id = request.data.get("user_id")
        year = request.data.get("year")

        if not all([user_id, year]):
            return Response(
                {"success": 0, "message": "필드 누락"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        try:
            user = People.objects.get(pk=user_id)
        except People.DoesNotExist:
            return Response(
                {"success": 0, "message": "존재하지 않는 사용자"},
                status=status.HTTP_404_NOT_FOUND,
            )

        diaries = Diary.objects.filter(
            user=user, marking=True, date__year=year
        ).select_related("emotion")

        emotions = []
        for diary in diaries:
            if diary.emotion:
                emotions.append(
                    {
                        "date": diary.date.strftime("%Y-%m-%d"),
                        "emotion": diary.emotion.emotion,
                    }
                )
        return Response(
            {"emotions": emotions},
            status=status.HTTP_200_OK,
        )


class MarkedMonthAPIView(APIView):
    permission_classes = [permissions.AllowAny]
    parser_classes = [parsers.JSONParser, parsers.FormParser]

    def post(self, request, *args, **kwargs):
        user_id = request.data.get("user_id")
        year = request.data.get("year")
        month = request.data.get("month")

        if not all([user_id, year, month]):
            return Response(
                {"success": 0, "message": "필드 누락"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        try:
            user = People.objects.get(pk=user_id)
        except People.DoesNotExist:
            return Response(
                {"success": 0, "message": "존재하지 않는 사용자"},
                status=status.HTTP_404_NOT_FOUND,
            )

        diaries = Diary.objects.filter(
            user=user, marking=True, date__year=year, date__month=month
        ).select_related("emotion")

        emotions = []
        for diary in diaries:
            if diary.emotion:
                emotions.append(
                    {
                        "date": diary.date.strftime("%Y-%m-%d"),
                        "emotion": diary.emotion.emotion,
                        "summary": diary.summary,
                    }
                )
        return Response(
            {"emotions": emotions},
            status=status.HTTP_200_OK,
        )
