import cv2
import numpy as np
import wave, struct
import os, glob
import sys
#sys.argv[1] is path to UCF_Frames

def getBlkIdxs(frame):
        h,w  = frame.shape
        blk00 = frame[:h//2,:w//2]
        blk01 = frame[:h//2,w//2:]
        blk10 = frame[h//2:,:w//2]
        blk11 = frame[h//2:,w//2:]
        
        blks = [blk00, blk01, blk10, blk11]
        sums = []
        means = []
        vars = []
        
        for idx, blk in enumerate(blks):
          #print(f"For block {idx}")
          sum = np.sum(blk)
          mean = np.mean(blk)
          var = np.var(blk)
          sums.append(sum); means.append(mean); vars.append(var)
        
        means = np.array(means)
        temp = np.argsort(means)
        ranks = np.empty_like(temp)
        ranks[temp] = np.arange(len(means))
        idxs = ranks
        
        return idxs

def rotateAlign(frame):

    idxs = getBlkIdxs(frame)
    
    #rotate image such that least idx is in top left block position
    if idxs[1] == 0: 
      out = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        
    elif idxs[2] == 0:
      out = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        
    elif idxs[3] == 0:
      out = cv2.rotate(frame, cv2.ROTATE_180)
        
    else:
      out = frame    
    idxs = getBlkIdxs(out)

    return out, idxs
  
def flipAndRotateAlign(frame):
    frame, idxs = rotateAlign(frame)
    
    #now do flip alignment
    
    if idxs[1] > idxs[2]:
      frame = cv2.flip(frame, 0)
      #1 - horizontal flip
      #0 - vertical flip
      #-1 - diagonal flip
      '''
      when followed by rotational alignment, hori and verti flips are the same
      diagonal flip is same as rotation by 180 degree
      '''
    idxs = getBlkIdxs(frame) #can comment this for faster compute, useless other than for display
    
    frame, idxs = rotateAlign(frame)

    return frame, idxs


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
                
                if sys.argv[3] == "90":
                    gray = cv2.rotate(gray, cv2.cv2.ROTATE_90_CLOCKWISE)
                if sys.argv[3] == "180":
                    gray = cv2.rotate(gray, cv2.cv2.ROTATE_180)

                out, _ = flipAndRotateAlign(gray)
                flat_gray = out.reshape(-1,)
                for value in flat_gray:
                        data = struct.pack('<h', value)
                        audioObj.writeframesraw(data)


if __name__ == "__main__":
    all_vid_path = sys.argv[1]
    for vid_path in os.listdir(all_vid_path):
            convert_video2audio1(all_vid_path+vid_path)
    
