from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods  ##
from django.views.decorators.http import require_POST  ##
from django.shortcuts import get_object_or_404
from django.http import Http404
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
@api_view(['GET'])
@permission_classes([IsAuthenticated])  # 인증된 사용자만 접근 가능(토큰 기반 인증 거침)
def getPeopleInfo(request):
    
    user = request.user  # 토큰 인증된 유저 정보
    person = get_object_or_404(People, id=user.id) #???????

    sharings = Sharing.objects.filter(owner=person, share_state="matched")

    if sharings.exists():
        serializer = SharingSerializer(sharings, many=True) #many=true로 해놓으면 반복문 효과
        sharing_data = serializer.data
    else:
        sharing_data = None
    
    #Response
    return Response({
        "user_id": person.id,
        "name": person.name,
        "sharing": sharing_data,
    })



# 이메일 수정
"""
이메일 수정 요청의 데이터포맷: {"user_id": 1, "new_email":"address@naver.com"}
"""

@csrf_exempt
@require_http_methods(["POST"])
def update_email(request):

    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"message": "잘못된 JSON 형식입니다."}, status=400)

    user_id = data.get("user_id")
    new_email = data.get("new_email")

    if not new_email or "@" not in new_email:
        return JsonResponse({"message": "유효한 이메일이 아닙니다."}, status=400)

    try:
        person = get_object_or_404(People, id=user_id)
    except Http404:
        return JsonResponse({"message": "해당 사용자를 찾을 수 없습니다."}, status=404)

    try:
        person.email = new_email
        person.save()
    except Exception as e:
        return JsonResponse({"error": f"이메일 수정 실패: {str(e)}"}, status=500)

    return JsonResponse({"message": "이메일이 성공적으로 수정되었습니다."}, status=200)


# 공개범위 수정
"""
보호자 연동 정보 블록 클릭 시 체크박스나 radio 버튼 등으로 공개범위 수정 가능하도록
버튼 체크하고, 저장 버튼 누르면 요청 오도록 한다.
이 기능은, 프론트에 연동된 사용자 블록이 뜬 상태에서, 그 블록을 클릭한 다음 수정이 이루어진다.
다대다 관계일 수 있으므로, 반드시 사용자 id와 보호자 id 모두를 요청 json 포맷에 포함시킨다.

요청 포맷: {"user_id": 1, "protector_id:=2, "공개범위": "partial"}
"""

@csrf_exempt
@require_http_methods(["POST"])
def update_showrange(request):
    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"message": "잘못된 JSON 형식입니다."}, status=400)

    user_id = data.get("user_id")
    protector_id = data.get("protector_id")
    new_range = data.get("공개범위")  # 'private' / 'partial' / 'full'

    if user_id is None or protector_id is None or new_range is None:
        return JsonResponse(
            {"message": "user_id, protector_id 또는 공개범위가 누락되었습니다."},
            status=400,
        )

    try:
        owner = People.objects.get(id=user_id)
        protector = People.objects.get(id=protector_id)
    except People.DoesNotExist:
        return JsonResponse(
            {"message": "사용자 또는 보호자를 찾을 수 없습니다."}, status=404
        )

    try:
        sharing = Sharing.objects.get(owner=owner, shared_with=protector)
    except Sharing.DoesNotExist:
        return JsonResponse({"message": "공유 관계가 존재하지 않습니다."}, status=404)

    # 공유 상태가 matched 인지 확인
    if sharing.share_state != "matched":
        return JsonResponse(
            {"message": "아직 연동이 완료되지 않은 보호자입니다."}, status=403
        )

    VALID_RANGE = ["private", "partial", "full"]
    if new_range not in VALID_RANGE:
        return JsonResponse(
            {"message": f"공개범위 값은 {VALID_RANGE} 중 하나여야 합니다."}, status=400
        )

    sharing.share_range = new_range
    sharing.save()

    return JsonResponse(
        {"message": "공개범위가 성공적으로 수정되었습니다."}, status=200
    )

# 아이디 검색
"""
요청 포맷: { "search_id": "ididid" }
응답 포맷: { "exists": true, "name": "사용자 이름", }
"""


@csrf_exempt
def search_user_by_id(request):
    if request.method == "POST":
        data = json.loads(request.body)
        string_id = data.get("string_id")
        try:
            user = People.objects.get(string_id=string_id)
            return JsonResponse(
                {"exists": True, "name": user.name, "found_id": user.string_id}
            )
        except People.DoesNotExist:
            return JsonResponse({"exists": False})


# 연동 요청 처리(string_id 기반)
"""
요청 JSON 예시: { "requester_id": 1, "target_string_id": "some_user_id", "relation": "father" }
"""


@csrf_exempt
@require_POST
def handle_sharing_request(request):
    data = json.loads(request.body)
    requester_id = data.get("requester_id")  # shared_with
    target_string_id = data.get("target_string_id")  # string_id로 owner 조회
    relation = data.get("relation")

    try:
        owner = People.objects.get(string_id=target_string_id)
    except People.DoesNotExist:
        return JsonResponse(
            {"error": "요청 대상 사용자가 존재하지 않습니다."}, status=404
        )

    try:
        shared_with = People.objects.get(id=requester_id)
    except People.DoesNotExist:
        return JsonResponse({"error": "요청자 사용자가 존재하지 않습니다."}, status=404)

    if Sharing.objects.filter(owner=owner, shared_with=shared_with).exists():
        return JsonResponse({"message": "이미 연동 요청을 보냈습니다."}, status=400)

    Sharing.objects.create(owner=owner, shared_with=shared_with, relation=relation)
    return JsonResponse({"message": "연동 요청을 보냈습니다."})


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



""" 
회원가입 로직!!(아이디 중복 불가)
회원가입 시 비밀번호 암호화 로직
비밀번호 수정 로직
회원가입 시 유효한 이메일인지지 검증 로직
로그인 로직(JWT이용용)
"""
