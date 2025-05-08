from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status, parsers, permissions
from people.models import People
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

        diary = Diary.objects.create(user=user, audio=audio)

        try:
            _ = datetime.datetime.strptime(date_str, "%Y-%m-%d").date()
        except ValueError:
            pass

        return Response(
            {"success": 1, "emotion": [0, 0, 0, 0, 0, 0, 0], "text": ""},
            status=status.HTTP_200_OK,
        )
