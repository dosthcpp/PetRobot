import socket 
from _thread import *
import os
import time
import requests, json
import check
from Control import *

##### 변수 설정 부분 #####
DATA_PATH = "./data/"
HOST = '127.0.0.1'
TCP_FORWARDED_HOST = '1.tcp.jp.ngrok.io:25327'
PORT = 8000
url = "https://kakaoi-newtone-openapi.kakao.com/v1/recognize"
key = 'my key'
headers = {
    "Content-Type": "application/octet-stream",
    "X-DSS-Service": "DICTATION",
    "Authorization": "KakaoAK " + key,
}
control=Control()

###########################

def threaded(client_socket, addr): 

    print('Connected by :', addr[0], ':', addr[1]) 
    # 클라이언트가 접속을 끊을 때 까지 반복합니다. 

    data_transferred = 0
    count = 0
    
    while True: 
        try:
            data = client_socket.recv(1024)
            cmd = ''

            if not data:
                print('Disconnected by ' + addr[0],':',addr[1])
                break

            SOUNDFILE = "./command-{}.wav".format(count)

            with open(SOUNDFILE, 'wb') as f:
                try:
                    while True:
                        if data[-4:] == b'DONE':
                            print("Done receiving")
                            break
                        f.write(data)
                        data = client_socket.recv(1024)
                        data_transferred += len(data)
                        
                except Exception as ex:
                    print(ex)

            print('파일 {} 받기 완료. 전송량 {}'.format(SOUNDFILE, data_transferred))
            
            name, accuracy = check.is_valid_voice(SOUNDFILE)

            print('This voice would be : ', name)
            print("Accuracy: ", accuracy)

            with open(SOUNDFILE, 'rb') as f:
                audio = f.read()
                try:
                    res = requests.post(url, headers=headers, data=audio)
                    result_json_string = res.text[res.text.index('{"type":"finalResult"'):res.text.rindex('}')+1]
                    result = json.loads(result_json_string)
                    cmd = result['value']
                    print(cmd)
                except Exception as ex:
                    print(ex)

            if cmd == '앞으로 가' or cmd == '앞으로가':
                for i in range(5):
                    control.forWard()
                control.stop()

            # if accuracy > 0.5:
            #     with open(SOUNDFILE, 'rb') as f:
            #         audio = f.read()
            #     try:
            #         res = requests.post(url, headers=headers, data=audio)
            #         print(res.text)
            #     except Exception as ex:
            #         print(ex)
            # else:
            #     print('목소리를 식별할 수 없습니다.')

            data_transferred = 0
            count = count + 1
            
            client_socket.send(b"DONE")

        except ConnectionResetError as e:
            print('Disconnected by ' + addr[0],':',addr[1])
            break
    client_socket.close() 

if __name__ == '__main__':
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((HOST, PORT)) 
    server_socket.listen() 

    print('server start')

    start_new_thread(os.system, ('/home/pi/ngrok tcp -region jp --remote-addr {} {}'.format(TCP_FORWARDED_HOST, PORT),))
    # 클라이언트가 접속하면 accept 함수에서 새로운 소켓을 리턴합니다.
    # 새로운 쓰레드에서 해당 소켓을 사용하여 통신을 하게 됩니다. 
    try:
        while True: 
            print('listening on PORT {}'.format(PORT))

            client_socket, addr = server_socket.accept() 
            start_new_thread(threaded, (client_socket, addr)) 
    except KeyboardInterrupt:
        server_socket.close()

