from django.urls import path
from . import views

urlpatterns=[
    path('',views.getPeopleInfo),
    path('update/email', views.update_email),
    #path('update/password',views.update_pw),
    path('delete/connection',views.disconnect_sharing),
]