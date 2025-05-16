from django.db import models


class test(models.Model):
    test = models.CharField(max_length=50, primary_key=True, null=False)


class People(models.Model):
    USER_TYPE_CHOICES = [
        ('user', '일반 사용자'),
        ('guardian', '보호자'),
    ]

    id = models.AutoField(primary_key=True)
    string_id = models.CharField(max_length=30, default='default')  # 외부 검색/로그인용 ID (중복 불가)
    name = models.CharField(max_length=100)
    email = models.EmailField(unique=True)
    password = models.CharField(max_length=128)
    user_type = models.CharField(max_length=10, choices=USER_TYPE_CHOICES, default='user') #디폴트는 기본 사용자(보호자는 protector). 보호자 계정으로 회원가입 시도중이라면, 보호자계정 회원가입 페이지 요청 들어오고, 계정 생성 시 보호자로 들어가게끔 

    def __str__(self):
        return self.name
    

class Sharing(models.Model):
    RANGE_CHOICES = [
        ('private', '비공개'),
        ('partial', '부분 공개'),
        ('full', '전체 공개'),
    ]

    STATE_CHOICES = [
        ('unmatched', '연동 대기'),
        ('matched', '연동 완료'),
    ]

    RELATION_CHOICES = [
    ('mother', '엄마'),
    ('father', '아빠'),
    ('daughter', '딸'),
    ('son', '아들'),
    ('doctor', '의료진'),
    ('default', '기타')
    ]

    id = models.AutoField(primary_key=True) #primary key 자동생성됨
    owner = models.ForeignKey(People, related_name='shared_items', on_delete=models.CASCADE)  # 공유하는 사람 (기본 사용자)
    shared_with = models.ForeignKey(People, related_name='request_shares', on_delete=models.CASCADE) 
    relation = models.CharField(max_length=20, choices=RELATION_CHOICES, default='default') # 프론트에서 radio 버튼 등을 만들어서, 각 옵션별로 가족, 의료진 등을 문자열로 요청에 담아 보내면 여기 저장됨
    share_range = models.CharField(max_length=10, choices=RANGE_CHOICES, default='private')  # default = 비공개
    share_state = models.CharField(max_length=10, choices=STATE_CHOICES, default='unmatched')  # default = 대기

