import cv2
import numpy as np
import wave, struct
import os, glob
import sys
#sys.argv[1] is path to UCF_Frames
#sys.argv[2] is resolution


def convert_video2audio1(vid_path):
        '''
        serialize video
        
        #ref https://www.tutorialspoint.com/read-and-write-wav-files-using-python-wave
        '''
        vidName  = vid_path.split("/")[-1]
        print(f"vid name = {vidName}")
    
        w,h = sys.argv[3].split("x")
        w = int(w)
        h = int(h)

        audioObj = wave.open(sys.argv[2]+vidName+".wav", 'wb')
        audioObj.setnchannels(1) # mono
        audioObj.setsampwidth(2)
        sampleRate = 128*128*25 #44100.0 # hertz
        audioObj.setframerate(sampleRate)

    
        for i in os.listdir(vid_path):
                frame = vid_path+"/"+i
                gray = cv2.imread(frame,0)
                gray = cv2.resize(gray, (w,h)) #to make it computationally pliable, and all videos to have a uniform spacial size
                #working on gray to be independent of color, maybe
                gray = cv2.resize(gray, (128,128)) #to make it computationally pliable, and all videos to have a uniform spacial size
                flat_gray = gray.reshape(-1,)
                for value in flat_gray:
                        data = struct.pack('<h', value)
                        audioObj.writeframesraw(data)


if __name__ == "__main__":
    all_vid_path = sys.argv[1]
    for vid_path in os.listdir(all_vid_path):
            convert_video2audio1(all_vid_path+vid_path)
    
