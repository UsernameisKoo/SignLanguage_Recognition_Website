import openai
import os
from dotenv import load_dotenv 
from pathlib import Path

# 상위 폴더의 .env 파일 경로
env_path = Path(__file__).resolve().parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

# 환경 변수에서 API 키 가져오기
api_key = os.getenv("OPENAI_API_KEY")
openai.api_key = api_key

def process_data(data):
    sentence = ""
    # 사용자 프롬프트 정의
    prompt = f"""
    나열된 단어를 다음 규칙을 순서대로 지켜서 하나의 문장으로 만드시오.
    만든 문장 외에 어떤 것도 작성하지 마시오: {data}

    1. 주어가 연속으로 있지 않는 경우, 주어를 기준으로 문장을 두개로 쪼갠다.
    - 나 집 사다 집 크다 -> 나는 집 사다. 집 크다. 
    (나-집은 연속이지만, 집-집은 연속이 아니므로. "나 집 / 집"으로 나뉜다. 이때 각 묶음의 첫번째 단어가 주어이다.)
    - 아이 울다 부모 달래주다 -> 아이 울다. 부모 달래주다.

    2. 한 문장 당 앞에 있는 것이 주어, 뒤에 있는 것이 목적어이며, 이에 따른 알맞은 조사를 채워넣는다.
    - 나 소연 -> 나는 소연입니다.
    - 나 책 읽는다 -> 나는 책을 읽는다.
    - 나 친구 선물 주다 -> 나는 친구에게 선물을 준다.

    3. '끝'은 과거를 나타내고, '중'은 진행 중임을 나타낸다. 
    - 밥 먹다 끝 가다 -> 밥을 먹고 간다.
    - 빵 먹다 중 -> 빵을 먹는 중이다.
    
    4. 필요시 품사를 바꿀 수 있다.
    - 친하다 사람 -> 친한 사람
    - 위험하다 -> 위험

    5. 1번에서 쪼개서 각각 번역한 문장을 하나로 합친다.
    - 나는 집을 샀다. 집이 크다. -> 나는 큰 집을 샀다
    - 아이는 운다. 부모는 달래준다. -> 아이가 울어서 부모가 달래줬다."""

    # GPT 모델 사용하여 응답 생성
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4", 
            messages=[
                {"role": "user", "content": prompt},
            ],
        )

        # 응답 가져오기 및 처리
        res = response['choices'][0]['message']['content']
        print("Chatbot Response:", res)
        sentence = res.strip('"')

    except openai.error.OpenAIError as e:
        print(f"OpenAI API 에러: {e}")

    return sentence

if __name__ == "__main__":
    # 테스트 데이터
    test_data = "나 집 사다 집 크다"
    result = process_data(test_data)
    print("결과:", result)
