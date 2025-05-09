from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.shortcuts import get_object_or_404
import json
from people.models import People, Sharing

# Create your views here.
@csrf_exempt
@require_http_methods(["POST"])
def getPeopleInfo(request):
    data=json.loads(request.body) #요청 바디에서 json 형식을 딕셔너리 자료형으로 변환환
    user_id=data.get("user_id") #key를 이용해 아이디 값 가져오기

    person=get_object_or_404(People,id=user_id)
    sharings = Sharing.objects.filter(owner=person)

    if sharings.exists():
        sharing_data = [
            {   
                "protector_id": sharing.shared_with.id,  #연동된 보호자 계정주 id
                "protector_name": sharing.shared_with.name, #연동된 보호자 계정주 이름
                "relation": sharing.relation,
                "공개범위": sharing.share_range,
            }
            for sharing in sharings
        ]
    else:
        sharing_data= None

    #JSON 포맷으로 데이터 반환
    return JsonResponse({
        "user_id": person.id,
        'name': person.name,
        "sharing": sharing_data
    })
   

@csrf_exempt
@require_http_methods(["POST"])
def update_email(request):
    """
이메일 수정 요청의 데이터포맷: {"user_id": 1, "new_email":"www@dfs.com"}
 """
    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"message": "잘못된 JSON 형식입니다."}, status=400)
    
    user_id = data.get("user_id")
    new_email=data.get("new_email")

    if not new_email or "@" not in new_email:
        return JsonResponse({"message": "유효한 이메일이 아닙니다."}, status=400)

    person = get_object_or_404(People, id=user_id)
    try:
        person.email = new_email
        person.save()
    except Exception as e:
        return JsonResponse({"error": f"이메일 수정 실패: {str(e)}"}, status=500)

    return JsonResponse({"message": "이메일이 성공적으로 수정되었습니다."}, status=200)

""" 
@csrf_exempt
@require_http_methods(["POST"])
def update_pw(request):
"""



"""
보호자 연동 정보 블록 클릭 시 체크박스나 radio 버튼 등으로 공개범위 수정 가능하도록
버튼 체크하고, 저장 버튼 누르면 요청 오도록
요청 포맷: {"user_id": 1, "protector_id:=2, "공개범위": 0}
"""
@csrf_exempt
@require_http_methods(["POST"])
def update_showrange(request):
    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"message": "잘못된 JSON 형식입니다."}, status=400)

    user_id = data.get("user_id")
    new_range = data.get("공개범위")

    if user_id is None or new_range is None:
        return JsonResponse({"message": "user_id 또는 공개범위가 누락되었습니다."}, status=400)
    
    person = get_object_or_404(People, id=user_id)

    


    




#disconnect_sharing(request)함수
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

        # 해당 관계 존재 여부 확인
        sharing = Sharing.objects.filter(owner_id=owner_id, shared_with_id=shared_with_id)

        if sharing.exists():
            sharing.delete()
            return JsonResponse({"message": "공유 관계가 성공적으로 삭제되었습니다."}, status=200)
        else:
            return JsonResponse({"message": "해당 공유 관계가 존재하지 않습니다."}, status=404)

    except json.JSONDecodeError:
        return JsonResponse({"message": "JSON 형식 오류"}, status=400)
    except Exception as e:
        return JsonResponse({"message": f"서버 오류: {str(e)}"}, status=500)