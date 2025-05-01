from django.db import models

class test(models.Model):
    test = models.CharField(max_length=50, primary_key=True, null=False)
