##### 화자인식 일반 머신러닝 코드 #####
import librosa
import librosa.display
import pyaudio #마이크를 사용하기 위한 라이브러리
import pickle
import wave
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
import joblib
import os
import os.path

##### 변수 설정 부분 #####
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100 #비트레이트 설정
CHUNK = int(RATE / 10) # 버퍼 사이즈 1초당 44100비트레이트 이므로 100ms단위
RECORD_SECONDS = 3 #녹음할 시간 설정
WAVE_OUTPUT_FILENAME = "output.wav"
SAVED_MODEL = 'saved_model.pkl'
DATA_PATH = "./data/"
train_data=[]#train_date 저장할 공강
train_label=[]#train_label 저장할 
test_data=[]#train_date 저장할 공강
test_label=[]#train_label 저장할 
##########################

def load_wave_generator(path): 
       
    batch_waves = []
    labels = []
    # input_width=CHUNK*6 # wow, big!!
    folders = os.listdir(path)
    #while True:
       # print("loaded batch of %d files" % len(files))
    for folder in folders:
        if not os.path.isdir(path) or not os.path.isdir(path+"/"+folder):
            continue
        # 폴더가 아니면 continue
        files = os.listdir(path+"/"+folder)
        print("Foldername :",folder,"-",len(files))
        # 폴더 이름과 그 폴더에 속하는 파일 갯수 출력
        for wav in files:
            if not wav.endswith(".wav"):
                continue
            else:
                global train_data, train_label
                    #전역변수를 사용하겠다.
                print("Filename :",wav)
                    #.wav 파일이 아니면 continue
                y, sr = librosa.load(path+"/"+folder+"/"+wav)
                mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=int(sr*0.01),n_fft=int(sr*0.02)).T
                if(len(train_data)==0):
                    train_data = mfcc
                    train_label = np.full(len(mfcc), int(folder))
                else:
                    train_data = np.concatenate((train_data, mfcc), axis = 0)
                    train_label = np.concatenate((train_label, np.full(len(mfcc),  int(folder))), axis = 0)
                    #print("mfcc :",mfcc.shape)

def record():
    ######## 음성 데이터를 녹음 해 저장하는 부분 ########

    p = pyaudio.PyAudio() # 오디오 객체 생성

    stream = p.open(format=FORMAT, # 16비트 포맷
                    channels=CHANNELS, #  모노로 마이크 열기
                    rate=RATE, #비트레이트
                    input=True,
                    frames_per_buffer=CHUNK) # CHUNK만큼 버퍼가 쌓인다.

    print("Start to record the audio.")

    frames = [] # 음성 데이터를 채우는 공간

    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)): 
        #지정한  100ms를 몇번 호출할 것인지 10 * 5 = 50  100ms 버퍼 50번채움 = 5초
        data = stream.read(CHUNK)
        frames.append(data)

    print("Recording is finished.")

    stream.stop_stream() # 스트림닫기
    stream.close() # 스트림 종료
    p.terminate() # 오디오객체 종료

    # WAVE_OUTPUT_FILENAME의 파일을 열고 데이터를 쓴다.
    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb') 
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()
                
if __name__ == '__main__':
    record()

    ######## 음성 데이터를 읽어와 학습 시키는 부분 ########
    clf = LogisticRegression()

    if not os.path.isfile(SAVED_MODEL):
        load_wave_generator(DATA_PATH)
        print("train_data.shape :", train_data.shape, type(train_data))
        print("train_label.shape :", train_label.shape, type(train_label))
        #print(mfcc[0])
        #print(train_label)
        ## 모델 데이터 트레이닝
        clf.fit(train_data,train_label)
        # X, y
        joblib.dump(clf, SAVED_MODEL)
    else:
        clf = joblib.load(SAVED_MODEL)

    # y, sr = librosa.load("./test_백도연.wav")

    # 녹음 파일
    y, sr = librosa.load(WAVE_OUTPUT_FILENAME)
    librosa.display.waveplot(y, sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=int(sr*0.01),n_fft=int(sr*0.02)).T

    y_test_estimated = clf.predict(mfcc)
    test_label = np.full(len(mfcc), 0)
    '''
    0 백도연
    1 유태경
    2 김재희
    '''
    # 정답률 구하기 
    ac_score = metrics.accuracy_score(y_test_estimated, test_label)
    print("정답률 =", ac_score)
    print(pd.value_counts(pd.Series(y_test_estimated)))
