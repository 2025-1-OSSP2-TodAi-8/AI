from django.contrib import admin
from django.urls import path
from . import views

urlpatterns = [
    path("day", views.emotions_day),
    path("month", views.emotions_month),
    path("year", views.emotions_year),
]
