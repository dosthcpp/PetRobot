import socket
import os
from os.path import exists
import time
import sys

count = 0
DURATION = 3
HOST = '1.tcp.jp.ngrok.io'
PORT = 25327

def record(SOUNDFILE):
    os.system('arecord -f S16_LE -D plughw:1,0 -d {} -r 16000 {}'.format(DURATION, SOUNDFILE))

def play():
    os.system("aplay -D sysdefault:CARD=seeed2micvoicec {}".format(SOUNDFILE))

if __name__ == '__main__':
    try:
        client_socket = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
        client_socket.connect((HOST, PORT))

        try:
            while True:
                SOUNDFILE = "./command-{}.wav".format(count)

                record(SOUNDFILE)
                # time.sleep(DURATION)
                # play()
                data_transferred = 0

                if not exists(SOUNDFILE):
                    print("no file")
                    sys.exit()
                with open(SOUNDFILE, 'rb') as f:
                    try:
                        data = f.read(1024)
                        while data:
                            data_transferred += client_socket.send(data)
                            data = f.read(1024)
                    except Exception as ex:
                        print(ex)
                print('파일을 전송하였습니다: command-{}.wav'.format(count))
                client_socket.send(b"DONE")
                count = count + 1

                # 다음명령 대기
                data = client_socket.recv(1024)
                if data[-4:] == b'DONE':
                    continue
        except KeyboardInterrupt:
            client_socket.close()

    except Exception as e:
        print(e)
