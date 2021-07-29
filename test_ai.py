import requests, json

url = "https://kakaoi-newtone-openapi.kakao.com/v1/recognize"
key = '996712050f93541a9843c267cfd65eb5'
headers = {
    "Content-Type": "application/octet-stream",
    "X-DSS-Service": "DICTATION",
    "Authorization": "KakaoAK " + key,
}

with open('./test_mother.wav', 'rb') as f:
    audio = f.read()
    try:
        res = requests.post(url, headers=headers, data=audio)
        result_json_string = res.text[res.text.index('{"type":"finalResult"'):res.text.rindex('}')+1]
        result = json.loads(result_json_string)
        print(result['value'])
    except Exception as ex:
        print(ex)