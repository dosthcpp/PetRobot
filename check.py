import tensorflow as tf
import librosa
import pyaudio
import numpy as np
import pandas as pd

##### 변수 설정 부분 #####
X_train = []#train_data 저장할 공간
X_test = []
Y_train = []
Y_test = []

def is_valid_voice(SOUNDFILE):

    y, sr = librosa.load(SOUNDFILE)

    X_test = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=int(sr*0.01),n_fft=int(sr*0.02)).T
    label = [0 for i in range(3)]
    Y_test = []
    for i in range(len(X_test)):
        Y_test.append(label)

    loaded_graph = tf.Graph()
    
    with tf.Session(graph=loaded_graph) as sess:
        saver = tf.train.import_meta_graph('./my_voice_model.meta')
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, tf.train.latest_checkpoint('./'))
        
        X = loaded_graph.get_tensor_by_name('X:0')
        Y = loaded_graph.get_tensor_by_name('Y:0')
        hypothesis = loaded_graph.get_tensor_by_name('h:0')
        keep_prob = loaded_graph.get_tensor_by_name('kp:0')
        
        correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        
        output_predict = pd.value_counts(pd.Series(sess.run(tf.argmax(hypothesis, 1), feed_dict={X: X_test, keep_prob:1})))
        output_accuracy = sess.run(accuracy, feed_dict={X: X_test, Y:Y_test, keep_prob:1})
        
        estimated = ["백도연", "유태경", "김재희"]

        name = estimated[pd.DataFrame(output_predict).index.values[0]]
        
        return name, output_accuracy

if __name__ == '__main__':
    # is_valid_voice('./output.wav')
    pass