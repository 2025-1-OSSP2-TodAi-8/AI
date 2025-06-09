from django.urls import path
from .views import *

urlpatterns = [
    path("record", record),
    path("get_record", get_record),
    path("marking", marking),
    path("marked_year", get_marked_year),
    path("marked_month", get_marked_month),
]
