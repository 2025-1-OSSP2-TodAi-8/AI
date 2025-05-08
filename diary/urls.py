from django.urls import path
from .views import *

urlpatterns = [
    path("record", RecordAPIView.as_view()),
    path("marking", MarkingAPIView.as_view()),
    path("marked_year", MarkedYearAPIView.as_view()),
    path("marked_month", MarkedMonthAPIView.as_view()),
]
