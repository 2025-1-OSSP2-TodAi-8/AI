from django.db import models
from django.contrib.auth.models import AbstractUser


class People(AbstractUser):
    USER_TYPE_CHOICES = [
        ("user", "일반 사용자"),
        ("guardian", "보호자"),
    ]

    GENDER_CHOICES = [
        ("male", "남성"),
        ("female", "여성"),
    ]

    # username 필드는 AbstractUser에 이미 존재하므로 별도 선언 불필요 (기존에 아이디로 쓴 string_id 대신 username을 아이디로 사용 예정정)
    name = models.CharField(max_length=100)
    email = models.EmailField(unique=True)
    user_type = models.CharField(max_length=10, choices=USER_TYPE_CHOICES, default="user")
    birthdate = models.DateField(null=True, blank=True)  # 생년월일
    gender = models.CharField(max_length=10, choices=GENDER_CHOICES)

    first_name = models.CharField(max_length=30, blank=True)  # blank=True 추가
    last_name = models.CharField(max_length=150, blank=True)  # blank=True 추가

    # 기본 USERNAME_FIELD는 'username'이라 별도 선언 필요 없음

    def __str__(self):
        return self.name



class Sharing(models.Model):
    RANGE_CHOICES = [
        ("private", "비공개"),
        ("partial", "부분 공개"),
        ("full", "전체 공개"),
    ]

    STATE_CHOICES = [
        ("unmatched", "연동 대기"),
        ("matched", "연동 완료"),
        ("rejected", "연동 거절"),
    ]
 
    # primary key 자동생성됨
    id = models.AutoField(primary_key=True)  
    # 공유하는 사람 (기본 사용자)
    owner = models.ForeignKey(
        People, related_name="shared_items", on_delete=models.CASCADE
    )  
    #공유 관계의 보호자
    shared_with = models.ForeignKey(
        People, related_name="request_shares", on_delete=models.CASCADE
    )
    #공개범위 (default = 비공개)
    share_range = models.CharField(
        max_length=10, choices=RANGE_CHOICES, default="private"
    )  
    #연동 상태 (default: unmatched /matched: 연결중/ rejected: 요청 거절됨)
    share_state = models.CharField(
        max_length=10, choices=STATE_CHOICES, default="unmatched"
    )  
