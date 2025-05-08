from django.db import models
from emotion.models import Emotion
from people.models import People


class Diary(models.Model):
    user = models.ForeignKey(People, on_delete=models.CASCADE)
    date = models.DateField(null=True)
    emotion = models.ForeignKey(Emotion, on_delete=models.SET_NULL, null=True)
    audio = models.FileField(upload_to="diary/audio/", null=True)
    summary = models.TextField(null=True)
    marking = models.BooleanField(default=False)

    def __str__(self):
        return f"{self.user.name}의 일기"
