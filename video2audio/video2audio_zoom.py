import cv2
import numpy as np
import wave, struct
import os, glob
import sys
#sys.argv[1] is path to UCF_Frames



def convert_video2audio1(vid_path):
        '''
        serialize video
        
        #ref https://www.tutorialspoint.com/read-and-write-wav-files-using-python-wave
        '''
        vidName  = vid_path.split("/")[-1]
        print(f"vid name = {vidName}")
       
        audioObj = wave.open(sys.argv[2]+vidName+".wav", 'wb')
        audioObj.setnchannels(1) # mono
        audioObj.setsampwidth(2)
        sampleRate = 128*128*25 #44100.0 # hertz
        audioObj.setframerate(sampleRate)

        for i in os.listdir(vid_path):
                frame = vid_path+"/"+i
                gray = cv2.imread(frame,0)
                #working on gray to be independent of color, maybe
                gray = cv2.resize(gray, (128,128)) #to make it computationally pliable, and all videos to have a uniform spacial size
         
         
                '''
                Manu
                param mag: magnification >= 1
                this function preserves aspect ratio
                '''
                mag = float(sys.argv[3])
                h1, w1 = gray.shape
                sqrtMag = np.sqrt(mag)
                h2 = int(h1//sqrtMag)
                w2 = int(w1//sqrtMag)

                h_cut = (h1-h2)//2
                w_cut = (w1-w2)//2
                
                gray_zoom = gray[h_cut:h1-h_cut,w_cut:w1-w_cut]
                gray_zoom = cv2.resize(gray_zoom, (w1,h1))
                

                flat_gray = gray_zoom.reshape(-1,)
                for value in flat_gray:
                        data = struct.pack('<h', value)
                        audioObj.writeframesraw(data)


if __name__ == "__main__":
    all_vid_path = sys.argv[1]
    for vid_path in os.listdir(all_vid_path):
            convert_video2audio1(all_vid_path+vid_path)
    
