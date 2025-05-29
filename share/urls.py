from django.urls import path
from . import views

urlpatterns=[
    path('search', views.search_user_by_id),
    path('sharing/request', views.handle_sharing_request),
]