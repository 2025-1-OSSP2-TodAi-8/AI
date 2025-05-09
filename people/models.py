from django.db import models


class test(models.Model):
    test = models.CharField(max_length=50, primary_key=True, null=False)


class People(models.Model):
    id = models.AutoField(primary_key=True)
    name = models.CharField(max_length=100)
    email = models.EmailField(unique=True)
    password = models.CharField(max_length=128)
    user_type = models.CharField(max_length=10, default='default') #디폴트는 기본 사용자(보호자는 protector). 보호자 계정으로 회원가입 시도중이라면, 보호자계정 회원가입 페이지 요청 들어오고, 계정 생성 시 보호자로 들어가게끔 

    def __str__(self):
        return self.name
    

class Sharing(models.Model):
    owner = models.ForeignKey(People, related_name='shared_items', on_delete=models.CASCADE)  # 공유하는 사람 (기본 사용자)
    shared_with = models.ForeignKey(People, related_name='received_shares', on_delete=models.CASCADE) 
    relation = models.CharField(max_length=100)  # 프론트에서 radio 버튼 등을 만들어서, 각 옵션별로 가족, 의료진 등을 문자열로 요청에 담아 보내면 여기 저장됨됨
    share_range = models.IntegerField(default=0)  # 0=비공개, 1=부분 공개, 2=전체공개

