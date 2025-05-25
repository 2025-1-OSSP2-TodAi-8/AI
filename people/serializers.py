from rest_framework import serializers
from .models import People, Sharing
#from django.contrib.auth.password_validation import validate_password #너무 짧은 비번인지 검증해주는 거라고 함. 이건 프론트에서 처리하고, 여기선 없어도 될듯듯

#회원가입입
class PeopleSignupSerializer(serializers.ModelSerializer):
    password = serializers.CharField(write_only=True, required=True)

    class Meta:
        model = People
        fields = ['username', 'password', 'name', 'email', 'user_type', 'birthdate', 'gender']

    def validate_username(self, value):
        if People.objects.filter(username=value).exists():
            raise serializers.ValidationError("이미 존재하는 아이디입니다.")
        return value

    def create(self, validated_data):
        user = People.objects.create_user(
            username=validated_data['username'],
            password=validated_data['password'],
            name=validated_data['name'],
            email=validated_data['email'],
            user_type=validated_data.get('user_type', 'user'),
            birthdate=validated_data.get('birthdate'),
            gender=validated_data.get('gender'),

            first_name=validated_data.get('first_name', ''),
            last_name=validated_data.get('last_name', ''),
            is_staff=False,
            is_active=True
        )
        return user

#마이페이지에서, 연동관계 있을 시 연동관계들 json 형식으로로
class SharingSerializer(serializers.ModelSerializer):
    protector_id = serializers.IntegerField(source='shared_with.id')
    protector_name = serializers.CharField(source='shared_with.name')
    공개범위 = serializers.CharField(source='share_range')

    class Meta:
        model = Sharing
        fields = ['protector_id', 'protector_name', '공개범위']