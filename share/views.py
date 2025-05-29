import json
from people.models import People, Sharing
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated


# 아이디 검색
"""
요청 포맷: { "search_id": "ididid" }
응답 포맷: { "exists": true, "name": "사용자 이름", }
"""
@api_view(['POST'])
@permission_classes([IsAuthenticated]) 
def search_user_by_id(request):
    search_id = request.data.get("search_id")

    if not search_id:
        return Response(
            {"message": "search_id 누락되었습니다."},
            status=status.HTTP_400_BAD_REQUEST
        )
    
    try:
        target = People.objects.get(username=search_id)
        return Response({
            "exists": True,
            "name": target.name,
            "found_id": target.username
        }, status=status.HTTP_200_OK)
    
    except People.DoesNotExist:
        return Response({"exists": False}, status=status.HTTP_200_OK)


# 연동 요청 처리
"""
요청 데이터포맷: 
{ 
    "target_user_id": "some_user_id", 
    "relation": "son" 
}
"""
@api_view(['POST'])
@permission_classes([IsAuthenticated])
def handle_sharing_request(request):
    data = request.data
    requester = request.user  # 보호자 계정으로 로그인한 사용자

    target_user_id = data.get("target_user_id")  # 공유 요청 대상 유저
    relation = data.get("relation")

    if not target_user_id or not relation:
        return Response(
            {"message": "target_user_id 또는 relation이 누락되었습니다."},
            status=status.HTTP_400_BAD_REQUEST
        )
    
    # 연동 요청 타겟이 DB에 존재하는지 확인
    try:
        target = People.objects.get(username=target_user_id)
    except People.DoesNotExist:
        return Response(
            {"message": "요청 대상인 타겟 사용자가 존재하지 않습니다."},
            status=status.HTTP_404_NOT_FOUND
        )
    
    # 이미 연동 요청 보낸 적 있는지 확인
    if Sharing.objects.filter(owner=target, shared_with=requester).exists():
        return Response(
            {"message": "이미 연동 요청을 보냈습니다."},
            status=status.HTTP_400_BAD_REQUEST
        )

    # 공유 요청 생성(Sharing 테이블에 튜플 추가 / share_state = unmatched 상태로 저장)
    Sharing.objects.create(owner=target, shared_with=requester, relation=relation)
    
    return Response(
        {"message": "연동 요청을 보냈습니다."},
        status=status.HTTP_201_CREATED
    )
