from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods  ##
from django.views.decorators.http import require_POST  ##

import json
from people.models import People, Sharing

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .serializers import PeopleSignupSerializer, SharingSerializer
from rest_framework.permissions import AllowAny
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated

#회원가입
class PeopleSignupView(APIView):
    permission_classes = [AllowAny]  # 회원가입은 누구나 접근 가능 DRF 클래스형 뷰에선 기본 권한이 IsAuthenticated이므로 명시적으로 AllowAny 지정

    def post(self, request):
        serializer = PeopleSignupSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response({"message": "회원가입 성공"}, status=status.HTTP_201_CREATED)
        
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    

# 마이페이지 기본화면
"""
요청 데이터포맷: 
- header에 액세스 토큰 정보
- json body에는 따로 포함할 데이터 없음.
"""
@api_view(['GET'])
@permission_classes([IsAuthenticated])  # 인증된 사용자만 접근 가능(토큰 기반 인증 거침)
def getPeopleInfo(request):
    
    person = request.user  # 토큰 인증된 유저 정보

    sharings = Sharing.objects.filter(owner=person, share_state="matched")

    if sharings.exists():
        serializer = SharingSerializer(sharings, many=True) #many=true로 해놓으면 반복문 효과
        sharing_data = serializer.data
    else:
        sharing_data = None
    
    return Response({
        "user_id": person.id,
        "name": person.name,
        "sharing": sharing_data,
    })


# 이메일 수정
"""
요청 데이터포맷: 
{ 
    "new_email":"address@naver.com"
}
"""
@api_view(['POST'])
@permission_classes([IsAuthenticated])
def update_email(request):
    new_email = request.data.get("new_email")

    if not new_email or "@" not in new_email:
        return Response({"message": "유효한 이메일이 아닙니다."}, status=status.HTTP_400_BAD_REQUEST)

    person = request.user  # People 모델의 인증된 사용자

    try:
        person.email = new_email
        person.save()
    except Exception as e:
        return Response({"error": f"이메일 수정 실패: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    return Response({"message": "이메일이 성공적으로 수정되었습니다."}, status=status.HTTP_200_OK)


#비밀번호 수정
'''
요청 데이터포맷: 
{
    "current_password": "old_password123",
    "new_password": "new_password456"
} 
'''
@api_view(['POST'])
@permission_classes([IsAuthenticated])
def update_password(request):
    current_password=request.data.get("current_password")
    new_password=request.data.get("new_password")

    if not current_password or not new_password:
        return Response({"messege": "현제 비밀번호와 새 비밀번호 모두 입력하세요."}, status=status.HTTP_400_BAD)
    
    person = request.user  # 인증된 사용자

    if not person.check_password(current_password):
        return Response({"message": "현재 비밀번호가 일치하지 않습니다."}, status=status.HTTP_400_BAD_REQUEST)

    try:
        person.set_password(new_password)
        person.save()
    except Exception as e:
        return Response({"error": f"비밀번호 수정 실패: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    return Response({"message": "비밀번호가 성공적으로 수정되었습니다."}, status=status.HTTP_200_OK)


# 공개범위 수정
"""
보호자 연동 정보 블록 클릭 시 체크박스나 radio 버튼 등으로 공개범위 수정 가능하도록
버튼 체크하고, 저장 버튼 누르면 요청 오도록 한다.
이 기능은, 프론트에 연동된 사용자 블록이 뜬 상태에서, 그 블록을 클릭한 다음 수정이 이루어진다.
다대다 관계일 수 있으므로, 반드시 사용자 id와 보호자 id 모두를 요청 json 포맷에 포함시킨다.

요청 데이터포맷:
{
  "protector_id": 2,
  "공개범위": "partial"
}
"""
@api_view(['POST'])
@permission_classes([IsAuthenticated])
def update_showrange(request):

    data=request.data
    owner = request.user

    protector_id = data.get("protector_id")
    new_range = data.get("공개범위")  # 'private' / 'partial' / 'full'

    if protector_id is None or new_range is None:
         return Response(
            {"message": "protector_id 또는 공개범위가 누락되었습니다."},
            status=status.HTTP_400_BAD_REQUEST
        )

    try:
        sharing = Sharing.objects.get(owner=owner, shared_with=protector_id)
    except Sharing.DoesNotExist:
        return Response(
            {"message": "공유 관계가 존재하지 않습니다."},
            status=status.HTTP_404_NOT_FOUND
        )

    # 공유 상태가 matched 인지 확인(기본적으로, 연동관계가 있어야 마이페이지에서 공개범위 수정 요청도 가능하나, 악의적인 요청 예방 위해 추가)
    if sharing.share_state != "matched":
        return Response(
            {"message": "아직 연동이 완료되지 않은 보호자입니다."},
            status=status.HTTP_403_FORBIDDEN
        )
    
    VALID_RANGE = ["private", "partial", "full"]
    if new_range not in VALID_RANGE:
        return Response(
            {"message": f"공개범위 값은 {VALID_RANGE} 중 하나여야 합니다."},
            status=status.HTTP_400_BAD_REQUEST
        )

    try:
        sharing.share_range = new_range
        sharing.save()
    except Exception as e:
        return Response(
            {"message": f"공개범위 수정 중 오류가 발생했습니다: {str(e)}"},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )

    return Response(
        {
            "message": "공개범위가 성공적으로 수정되었습니다.",
            "updated_range": new_range,
            "protector_id": protector_id
        },
        status=status.HTTP_200_OK
    )



# 연동 요청 수락 처리
"""
요청 JSON 예시: { "share_id": 10 }  #sharing 테이블의 primary키
월별 데이터 조회화면(메인화면) API에, sharing 테이블의 튜플이 unmatched인 경우 notification 정보 같이 주는 코드 추가해야함. 
"""


@csrf_exempt
@require_POST
def accept_sharing_request(request):
    data = json.loads(request.body)
    share_id = data.get("share_id")

    try:
        share = Sharing.objects.get(id=share_id)
        share.share_state = "matched"
        share.save()
        return JsonResponse({"message": "연동 요청을 수락했습니다."})
    except Sharing.DoesNotExist:
        return JsonResponse({"error": "해당 요청이 존재하지 않습니다."}, status=404)


# disconnect_sharing(request)함수
"""
연결끊기 요청의 데이터포맷: {"user_id": 1, "shared_with":"연결 사용자 아이디?"}
"""

@csrf_exempt
@require_http_methods(["POST"])
def disconnect_sharing(request):
    try:
        data = json.loads(request.body)
        owner_id = data.get("user_id")
        shared_with_id = data.get("shared_with")

        if not owner_id or not shared_with_id:
            return JsonResponse({"message": "요청 데이터가 불완전합니다."}, status=400)

        # matched 상태의 공유 관계만 삭제
        sharing = Sharing.objects.filter(
            owner_id=owner_id, shared_with_id=shared_with_id, share_state="matched"
        )

        if sharing.exists():
            sharing.delete()
            return JsonResponse(
                {"message": "공유 관계가 성공적으로 삭제되었습니다."}, status=200
            )
        else:
            return JsonResponse(
                {"message": "해당 공유 관계가 존재하지 않습니다."}, status=404
            )

    except json.JSONDecodeError:
        return JsonResponse({"message": "JSON 형식 오류"}, status=400)
    except Exception as e:
        return JsonResponse({"message": f"서버 오류: {str(e)}"}, status=500)


