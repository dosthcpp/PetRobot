import os
import argparse
import subprocess

from pydub import AudioSegment

formats_to_convert = ['.mp3']

for (dirpath, dirnames, filenames) in os.walk("./"):
    for filename in filenames:
        if filename.endswith(tuple(formats_to_convert)):
            subprocess.call(['ffmpeg', '-i', filename, '{0}.wav'.format(filename.split('.')[0])])
        #     filepath = dirpath + '/' + filename
        #     (path, file_extension) = os.path.splitext(filepath)
        #     file_extension_final = file_extension.replace('.', '')
        #     try:
        #         track = AudioSegment.from_file(filepath,
        #                 file_extension_final)
        #         wav_filename = filename.replace(file_extension_final, 'wav')
        #         wav_path = dirpath + '/' + wav_filename
        #         print('CONVERTING: ' + str(filepath))
        #         file_handle = track.export(wav_path, format='wav')
        #         os.remove(filepath)
        #     except:
        #         print("ERROR CONVERTING " + str(filepath))