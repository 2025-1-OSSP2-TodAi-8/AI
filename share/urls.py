from django.urls import path
from . import views

app_name='share'

urlpatterns=[
    path('/search', views.search_user_by_id),
    path('/sharing/request', views.handle_sharing_request),
]