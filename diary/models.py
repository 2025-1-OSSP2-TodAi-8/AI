from django.db import models
from emotion.models import Emotion
from people.models import People


class Diary(models.Model):
    user = models.ForeignKey(People, on_delete=models.CASCADE)
    emotion = models.ForeignKey(Emotion, on_delete=models.SET_NULL, null=True)
    audio = models.FileField(upload_to="diary/audio/")
    summary = models.TextField()

    def __str__(self):
        date_str = self.emotion.date.strftime("%Y-%m-%d")
        return f"{self.user.name} — {date_str}"
