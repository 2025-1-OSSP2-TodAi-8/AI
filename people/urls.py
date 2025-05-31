from django.urls import path
from . import views
from .views import PeopleSignupView  
from .views import LogoutView #추기
from rest_framework_simplejwt.views import TokenObtainPairView, TokenRefreshView 

urlpatterns=[
    path('',views.getPeopleInfo),
    path('update/email', views.update_email),
    path('delete/connection',views.disconnect_sharing),
    path('update/showrange', views.update_showrange),

    path('update/password', views.update_password),

    path('sharing/accept', views.accept_sharing_request),

    path('signup', PeopleSignupView.as_view(), name='signup'), #회원가입
    path('signin', TokenObtainPairView.as_view(), name='token_obtain_pair'), #로그인
    path('signin/refresh', TokenRefreshView.as_view(), name='token_refresh'), #리프레시 토큰 요청
    path('logout', LogoutView.as_view(), name='logout' ), #로그아웃
]
