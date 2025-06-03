from people.models import People, Sharing
from diary.models import Diary
from emotion.models import Emotion

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .serializers import PeopleSignupSerializer, SharingSerializer, RequestedSharingSerializer
from rest_framework.permissions import AllowAny
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework_simplejwt.tokens import RefreshToken, TokenError

from rest_framework_simplejwt.views import TokenObtainPairView
from .serializers import CustomTokenObtainPairSerializer

#회원가입
class PeopleSignupView(APIView):
    permission_classes = [AllowAny]  # 회원가입은 누구나 접근 가능 DRF 클래스형 뷰에선 기본 권한이 IsAuthenticated이므로 명시적으로 AllowAny 지정

    def post(self, request):
        serializer = PeopleSignupSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response({"message": "회원가입 성공"}, status=status.HTTP_201_CREATED)
        
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    

#로그인
class CustomTokenObtainPairView(TokenObtainPairView):
    serializer_class = CustomTokenObtainPairSerializer


# 마이페이지 기본화면
"""
요청 데이터포맷: 
- header에 액세스 토큰 정보
- json body에는 따로 포함할 데이터 없음.
"""
@api_view(['GET'])
@permission_classes([IsAuthenticated])  # 인증된 사용자만 접근 가능(토큰 기반 인증 거침)
def getPeopleInfo(request):

    person = request.user  

    #연동이 된 보호자 계정들 조회
    sharings = Sharing.objects.filter(owner=person, share_state="matched")
    if sharings.exists():
        serializer = SharingSerializer(sharings, many=True) #many=true로 해놓으면 반복문 효과
        sharing_data = serializer.data
    else:
        sharing_data = None
    
    # 알림용 - 요청 들어온 정보 조회
    requested_sharings=Sharing.objects.filter(owner=person, share_state="unmatched")
    if requested_sharings.exists():
        serializer = RequestedSharingSerializer(requested_sharings, many=True) #many=true로 해놓으면 반복문 효과
        requested_sharing_data = serializer.data
    else:
        requested_sharing_data = None

    return Response({
        "user_id": person.id,
        "name": person.name,
        "sharing": sharing_data,
        "notification": requested_sharing_data
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
        return Response({"message": "현재 비밀번호와 새 비밀번호 모두 입력하세요."}, status=status.HTTP_400_BAD_REQUEST)
    
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
        sharing = Sharing.objects.get(owner=owner, shared_with=protector_id, share_state="matched")
    except Sharing.DoesNotExist:
        return Response(
            {"message": "공유 관계가 존재하지 않습니다."},
            status=status.HTTP_404_NOT_FOUND
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
            {"message": "search_id가가 누락되었습니다."},
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
    "target_user_id": "hongildong",
}
"""
@api_view(['POST'])
@permission_classes([IsAuthenticated])
def handle_sharing_request(request):
    data = request.data
    requester = request.user  # 보호자 계정으로 로그인한 사용자

    target_user_id = data.get("target_user_id")  # 공유 요청 대상 유저
    
    if not target_user_id:
        return Response(
            {"message": "target_user_id가 누락되었습니다."},
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
    if Sharing.objects.filter(owner=target, shared_with=requester, share_state="unmatched").exists():
        return Response(
            {
                "message": "이미 연동 요청을 보냈습니다.",
                "target_user_id": target.id
            },
            status=status.HTTP_400_BAD_REQUEST
        )
    
    if Sharing.objects.filter(owner=target, shared_with=requester, share_state="matched").exists():
        return Response(
            {
                "message": "이미 연동이 완료된 사용자입니다.",
                "target_user_id": target.id
            },
            status=status.HTTP_400_BAD_REQUEST
        )

    # 공유 요청 생성(Sharing 테이블에 튜플 추가 / share_state = unmatched 상태로 저장)
    Sharing.objects.create(owner=target, shared_with=requester)
    
    return Response(
        {
            "message": "연동 요청을 보냈습니다.",
            "target_user_id": target.id
        },
        status=status.HTTP_201_CREATED
    )

# 연동 요청 수락 처리
"""
요청 JSON 예시: 
{ 
"sharing_id": 10 
"action": accept OR reject
}  #sharing 테이블의 primary키

마이페이지 API에, sharing 테이블의 튜플 중 현재 로그인한 사용자의 아이디이고, unmatched인 경우 notification 정보 같이 주는 코드 추가해야함. 
unmatched이면, 메인화면 뷰에서 응답 json에 notification 정보 추가해야함
"""
@api_view(['POST'])
@permission_classes([IsAuthenticated])
def accept_sharing_request(request):
    sharing_id = request.data.get("sharing_id")
    action=request.data.get("action")

    if sharing_id is None or action is None:
        return Response(
            {"message": "sharing_id와 action 필드 중 누락된 데이터가 있습니다."},
            status=status.HTTP_400_BAD_REQUEST
        )
    
    if action not in ["accept", "reject"]:
        return Response(
            {"message": "action은 'accept' 또는 'reject'만 가능합니다."},
            status=status.HTTP_400_BAD_REQUEST
        )
    
    try:
        share = Sharing.objects.get(id=sharing_id, owner=request.user, share_state="unmatched")

        if action == "accept":
            share.share_state = "matched"
        elif action == "reject":
            share.share_state = "rejected"

        share.save()
        return Response({"message": f"연동 요청이 {action}되었습니다."})
    
    except Sharing.DoesNotExist:
        return Response({"message": "해당 연동 요청이 존재하지 않습니다."}, status=status.HTTP_404_NOT_FOUND)


# disconnect_sharing(request)함수
"""
연결끊기 요청의 데이터포맷: {"shared_with":"연결 보호자 기본키(마이페이지의 연동 정보 표시할 때 데이터 포함되어 있음)"}
"""
@api_view(['POST'])
@permission_classes([IsAuthenticated])
def disconnect_sharing(request):
    try:
        owner=request.user
        shared_with = request.data.get("shared_with")

        if not shared_with:
            return Response({"message": "shared_with 필드가 필요합니다."}, status=status.HTTP_400_BAD_REQUEST)

        Sharing_data = Sharing.objects.filter(
            owner=owner,
            shared_with_id=shared_with,
            share_state="matched"
        )

        if Sharing_data.exists():
            # 삭제 대신 상태 변경(거절)
            Sharing_data.update(share_state="rejected")
            return Response({"message": "연동 끊기 성공"}, status=status.HTTP_200_OK)
        else:
            return Response({"message": "연동된 공유 관계가 존재하지 않습니다."}, status=status.HTTP_404_NOT_FOUND)

    except Exception as e:
        return Response({"message": f"서버 오류: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


#로그아웃
'''
요청 데이터포맷:
{
    "refresh": "리프레시토큰문자열"
}
'''
class LogoutView(APIView):
    permission_classes = [IsAuthenticated]
    def post(self, request):
        try:
            refresh_token=request.data["refresh"] 

            token=RefreshToken(refresh_token) #토큰 객체로 변환
            token.blacklist()

            return Response(
                {"message": "로그아웃 완료"},
                status=status.HTTP_205_RESET_CONTENT
            )

        except KeyError:
            return Response({"message": "Refresh 토큰이 필요합니다."}, status=status.HTTP_400_BAD_REQUEST)
        except TokenError as e:
            return Response({"message": f"유효하지 않은 토큰입니다: {str(e)}"}, status=status.HTTP_400_BAD_REQUEST)




EMOTION_LABELS = ["행복", "슬픔", "놀람", "화남", "혐오", "공포", "중립"]

#보호자 페이지_공개범위 0
'''
요청 데이터포맷:
{
    "user_id" : 1,
	"year" : 2024, 
	"month" : 5 
}
'''
@api_view(["POST"])
@permission_classes([IsAuthenticated])
def emotions_month_for_protector(request):
    protector = request.user  # 보호자(로그인한 사용자)

    year = request.data.get("year")
    month = request.data.get("month")
    user_id = request.data.get("user_id")  # 감정 기록 조회 대상

    if not all([year, month, user_id]):
        return Response({"error": "year, month, user_id 모두 필요합니다."}, status=400)

    try:
        year = int(year)
        month = int(month)
        user_id = int(user_id)
        if not (1 <= month <= 12):
            raise ValueError
    except (ValueError, TypeError):
        return Response(
            {"error": "year, month, user_id 유효한 정수여야 하며, month는 1~12여야 합니다."},
            status=400
        )

    # 연결된 사용자 가져오기
    try:
        target_user = People.objects.get(id=user_id)
    except People.DoesNotExist:
        return Response({"error": "연결된 사용자가 존재하지 않습니다."}, status=404)

    # 연동 상태 확인
    sharing = Sharing.objects.filter(
        owner=target_user,
        shared_with=protector,
        share_state="matched",
    ).first()
    if not sharing:
        return Response({"error": "연동된 사용자가 아닙니다."}, status=403)
    
    allowed = ["private", "partial", "full"]
    if sharing.share_range not in allowed:
        return Response({"error": "해당 사용자의 페이지는 비공개입니다."}, status=403)


    # 감정 기록 조회
    diaries = Diary.objects.filter(
        user=target_user, date__year=year, date__month=month
    ).order_by("date")

    emotions = []
    for d in diaries:
        probs = d.emotion or []
        if len(probs) == len(EMOTION_LABELS):
            idx = max(range(len(probs)), key=lambda i: probs[i])
            label = EMOTION_LABELS[idx]
        else:
            label = ""

        emotions.append({"date": d.date.isoformat(), "emotion": label})

    return Response({
        "user_name": target_user.name,
        "user_id": user_id,
        "emotions": emotions
    }, status=200)


#보호자 페이지_공개범위 1_ 즐겨찾기
@api_view(["POST"])
@permission_classes([IsAuthenticated])
def get_marked_month_for_protector(request):
    protector = request.user
    user_id = request.data.get("user_id")
    year = request.data.get("year")
    month = request.data.get("month")

    if user_id is None or year is None or month is None:
        return Response({"error": "user_id, year, month 모두 필요합니다."}, status=400)

    try:
        user_id = int(user_id)
        year = int(year)
        month = int(month)
    except (ValueError, TypeError):
        return Response({"error": "user_id, year, month는 정수여야 합니다."}, status=400)

    # 타겟 사용자 존재 여부 확인
    try:
        target_user = People.objects.get(id=user_id)
    except People.DoesNotExist:
        return Response({"error": "해당 사용자가 존재하지 않습니다."}, status=404)


    # 연동 상태 확인
    sharing = Sharing.objects.filter(
        owner=target_user,
        shared_with=protector,
        share_state="matched",
    ).first()
    if not sharing:
        return Response({"error": "연동된 사용자가 아닙니다."}, status=403)
    
    allowed = ["partial", "full"]
    if sharing.share_range not in allowed:
        return Response({"error": "해당 페이지는 비공개 처리 되었습니다."}, status=403)


    # Diaries 조회 (target_user의 marking=True, year, month 조건)
    diaries = Diary.objects.filter(
        user=target_user,
        marking=True,
        date__year=year,
        date__month=month,
    ).order_by("date")

    emotions = []
    for d in diaries:
        probs = d.emotion or []
        if len(probs) == len(EMOTION_LABELS):
            idx = max(range(len(probs)), key=lambda i: probs[i])
            label = EMOTION_LABELS[idx]
        else:
            label = ""
        emotions.append(
            {
                "date": d.date.isoformat(), 
                "emotion": label,
                "summary": d.summary
            })

    return Response({"emotions": emotions}, status=200)