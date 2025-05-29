from django.urls import path
from . import views
from .views import PeopleSignupView  # 추가
from rest_framework_simplejwt.views import TokenObtainPairView, TokenRefreshView  # 추가

<<<<<<< HEAD
urlpatterns=[
    path('',views.getPeopleInfo),
    path('/update/email', views.update_email),
    path('/delete/connection',views.disconnect_sharing),
    path('/update/showrange', views.update_showrange),

    path('/search', views.search_user_by_id),
    path('/sharing/request', views.handle_sharing_request),
    path('/sharing/accept', views.accept_sharing_request),

    path('/signup', PeopleSignupView.as_view(), name='signup'), #회원가입
    path('/login', TokenObtainPairView.as_view(), name='token_obtain_pair'), #로그인
    path('/token/refresh', TokenRefreshView.as_view(), name='token_refresh'), #리프레시 토큰 요청
]
=======
urlpatterns = [
    path("", views.getPeopleInfo),
    path("update/email", views.update_email),
    path("delete/connection", views.disconnect_sharing),
    path("update/showrange", views.update_showrange),
    path("search", views.search_user_by_id),
    path("sharing/request", views.handle_sharing_request),
    path("sharing/accept", views.accept_sharing_request),
    path("signup", PeopleSignupView.as_view(), name="signup"),  # 회원가입
    path("signin", TokenObtainPairView.as_view(), name="token_obtain_pair"),  # 로그인
    path("signin/refresh", TokenRefreshView.as_view(), name="token_refresh"),  # 로그인
]
>>>>>>> b61e85f80254f8f38aff285317e25178d217b6ac
