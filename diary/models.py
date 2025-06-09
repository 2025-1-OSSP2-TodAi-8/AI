from django.db import models
from django.contrib.postgres.fields import ArrayField
from people.models import People


class Diary(models.Model):
    user = models.ForeignKey(People, on_delete=models.CASCADE)
    date = models.DateField(null=True)
    emotion = ArrayField(
        base_field=models.FloatField(),
        size=6,
        default=list,
        verbose_name="감정 비율",
        help_text="[기쁨, 슬픔, 분노, 놀람, 공포, 혐오] 순서로 0~1 사이의 확률 리스트",
    )
    audio = models.FileField(upload_to="diary/audio/", null=True)
    summary = models.TextField(null=True)
    marking = models.BooleanField(default=False)

    def __str__(self):
        return f"일자 : {self.date} || 작성자 : {self.user.name}"
