from django.db import models


class test(models.Model):
    test = models.CharField(max_length=50, primary_key=True, null=False)


class People(models.Model):
    id = models.AutoField(primary_key=True)
    name = models.CharField(max_length=100)
    email = models.EmailField(unique=True)
    password = models.CharField(max_length=128)

    def __str__(self):
        return self.name
