import numpy as np
import wave, struct
import os, glob
import sys
import cv2
#sys.argv[1] is path to UCF_Frames
#sys.argv[2] is output wav path
#sys.argv[3] is watermark image

def addWatermark(frame, wmImg, wmResize= (25, 25), pos=(0.85,0.01)):
        '''
        adds watermark to frame
        param frame: input frame
        param wmImg: watermark image
        param wmResize: watermark resize param (width, height)
        param pos: position of watermark img (y,x), range from 0 to 1
        '''
        frame = frame[:,:,:3] #remove alfa channel
        h1,w1,_ = frame.shape
        h2,w2,_ = wmImg.shape
        
        wmImg = cv2.resize(wmImg, (wmResize[0], wmResize[1]))
        h2,w2,_ = wmImg.shape
        wmImg = wmImg[:,:,:3] #remove alfa channel
        
        #hmargin = 10;  wmargin = 10 #margins, not needed
        locY = int(pos[0]*h1); locX = int(pos[1]*w1)
        
        #rloc= [h1-hmargin-h2:h1-hmargin, w1-wmargin-w2:w1-wmargin]
        #roi = frame[h1-hmargin-h2:h1-hmargin, w1-wmargin-w2:w1-wmargin]
        roi = frame[locY:locY+h2, locX:locX+w2]
        
        #chroma key
        roi = np.where(wmImg[:,:,:] == np.array([0,0,0]), roi, wmImg)
        
        wmFrame = frame.copy()
        #wmFrame[h1-hmargin-h2:h1-hmargin, w1-wmargin-w2:w1-wmargin,:] = roi
        wmFrame[locY:locY+h2, locX:locX+w2] = roi
        
        return wmFrame

def convert_video2audio1(vid_path,watermark):
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

                frame = cv2.imread(frame)

                if sys.argv[4] == "tl":
                    frame = addWatermark(frame, watermark, pos=(0.005,0.005))
                if sys.argv[4] == "bl":
                    frame = addWatermark(frame, watermark, pos=(0.75,0.005))
                if sys.argv[4] == "tr":
                    frame = addWatermark(frame, watermark, pos=(0.005,0.75))
                if sys.argv[4] == "br":
                    frame = addWatermark(frame, watermark, pos=(0.75,0.75))

                #working on gray to be independent of color, maybe
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.resize(gray, (128,128)) #to make it computationally pliable, and all videos to have a uniform spacial size
               # cv2.imshow("lol",gray)
               # cv2.waitKey()
                flat_gray = gray.reshape(-1,)
                for value in flat_gray:
                        data = struct.pack('<h', value)
                        audioObj.writeframesraw(data)


if __name__ == "__main__":
    all_vid_path = sys.argv[1]
    watermark = cv2.imread(sys.argv[3])
    for vid_path in os.listdir(all_vid_path):
            convert_video2audio1(all_vid_path+vid_path, watermark)
    
