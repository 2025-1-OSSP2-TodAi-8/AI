from django.db import models

# Create your models here.
# emotion/models.py

from django.db import models
from people.models import People


class Emotion(models.Model):
    user = models.ForeignKey(People, on_delete=models.CASCADE)
    date = models.DateField()  # 기록 날짜
    emotion = models.CharField(
        max_length=10
    )  # 감정 : 놀람, 화남, 행복, 슬픔, 혐오, 공포
    created_at = models.DateTimeField(auto_now_add=True, verbose_name="등록일시")
    updated_at = models.DateTimeField(auto_now=True, verbose_name="수정일시")

    def __str__(self):
        return f"유저: {self.user.id}, 일자: {self.date}, 감정: {self.emotion}"
