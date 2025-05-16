from django.urls import path
from . import views

urlpatterns=[
    path('',views.getPeopleInfo),
    path('update/email', views.update_email),
    #path('update/password',views.update_pw),
    path('delete/connection',views.disconnect_sharing),
    path('update/showrange', views.update_showrange),

    path('search/', views.search_user_by_id),
    path('sharing/request/', views.handle_sharing_request),
    #path('api/main/', views.main_page, name='main_page'),
    path('sharing/accept/', views.accept_sharing_request),
]