import cv2
import numpy as np
import wave, struct
import os, glob
import sys
#sys.argv[1] is path to UCF_Frames
from scipy.fftpack import dct, idct
import pdb
#from helper import getDistance #it is part of project 2, copy if you need it
import pandas as pd
import os




#ref
def dct2(x):
        return dct( dct( x, axis=0, norm='ortho' ), axis=1, norm='ortho' )

def idct2(x):
        return idct( idct( x, axis=0 , norm='ortho'), axis=1 , norm='ortho')
# 2D DCT and IDCT

def get_zigzag_matrix(M, verbose=False): #gets the zigzag matriz
    '''
    Generates zigzag matrix, also stores those indices in the zigzag order.
    Makes it a lot easier later.
    :param M: the order of the matrix
    :return:
    '''

    mat = np.ones((M,M), np.int) * -1
    count = 0
    total_counts = M*M
    ordIdx = [] #ordered indices
    #i = 0; j = 0;


    #create zigzag
    for k in range(M):
        for i in range(k+1):
            #print(f"k = {k}, i = {i}, k-i = {k-i}")
            if k%2 == 0:
                mat[i,k-i] = count
                ordIdx.append((i,k-i))
                count += 1
            else:
                mat[k-i,i] = count
                ordIdx.append((k-i,i))
                count += 1
    if verbose == True: print(f"zigzag mat1 = \n{mat}")
    for k in range(M, 2*(M-1)+1):
        for i in range(k-M+1, M):
            #print(f"k = {k}, i = {i}, k-i = {k-i}")
            #pdb.set_trace()
            if k%2 == 0:
                mat[i,k-i] = count
                ordIdx.append((i,k-i))
                count += 1
            else:
                mat[k-i,i] = count
                ordIdx .append((k-i,i))
                count += 1

    if verbose == True:
        print(f"zigzag mat = \n{mat}")
        print(f"Oredered Indices = {ordIdx}")

    return mat, ordIdx

def intQuantizeDctBlk(dBlk, blk_size=None, verbose=False):

    '''
    Author: Manu Ramesh

    Takes in DCT block and integer quantizes it according to JPEG quantization table
    for block sizes other than 8x8, the quantization matrix is rescaled and interpolated (bilinear) using cv2 resize fn
    param dBlk: dct block
    param blk_size: block size (if you want to give)
    return: integer dct block
    '''

    if blk_size == None:
      h, w = dBlk.shape
      blk_size = h # = w, same

    Q_mat_JPEG = np.array([  [16,  11,  10,  16,  24,  40,  51,  61],
    [12,  12,  14,  19,  26,  58,  60,  55],
    [14,  13,  16,  24,  40,  57,  69,  56],
    [14,  17,  22,  29,  51,  87,  80,  62],
    [18,  22,  37,  56,  68, 109, 103,  77],
    [24,  35,  55,  64,  81, 104, 113,  92],
    [49,  64,  78,  87, 103, 121, 120, 101],
    [72,  92,  95,  98, 112, 100, 103,  99]])

    int_dBlk = np.zeros(dBlk.shape, np.int)

    #resize the default quantization matrix like how you resize images --> bilinear interpolation!
    Q_mat = cv2.resize(Q_mat_JPEG.astype(float), (blk_size, blk_size)).astype(int)

    maxVal = 256*blk_size #this is the max possible value that can be acheived by each dct coeff
    #https://docs.scipy.org/doc/scipy/reference/generated/scipy.fftpack.dct.html substituite type 2 dct fromula from this link and check
    #max dc val = sum of all elements times the normalizing factor
    #there is always dc in images as no value can go -ve --> no -ve light, background
    #dct vals of non dc coeffs can be -ve --> as per formula -v max val to + max val but check if it is -maxval/2 to +maxval/2

    for i in range(blk_size):
      for j in range(blk_size):
        if i == 0 and j  == 0:
          bins = np.linspace(0,maxVal,Q_mat[i,j]).astype(int)
        else:

          bins = np.linspace(0,maxVal,Q_mat[i,j]//2).astype(int)

        #taking sign outside to make sure that 0 is always included as a bin element
        qval = bins[np.digitize(abs(dBlk[i,j]), bins=bins)]*np.sign(dBlk[i,j])

        if verbose == True: print(f"\nbins = {bins}")
        if verbose == True: print(f"val in = {dBlk[i,j]}, valOut = {qval}")
        int_dBlk[i,j] = qval

    return int_dBlk


def dctBlockWise(gray, blkSize, K, intQ = True, verbose=False):
    '''
    Takes block wise dct, performs integer quantization and retains K coefficinents in each block
    param gray: input grayscale image
    param blkSize: size of DCT block
    param K: number of coefficients to be used
    param intQ: turns integer quantization of dct block on
    return: quantized image
    '''
    h, w = gray.shape
    outImg = np.zeros_like(gray)
    dctFrame = np.zeros(gray.shape, np.float)

    mask, ordIdx = get_zigzag_matrix(blkSize)
    #mask = np.where(mask>K, 0, 255).astype(np.uint8) #push all other values to 0 #can use this statement with cv2 bitwise anding
    if verbose == True: print(f"Mask = \n{mask}")

    allRetainedCoeffs = [] #retained coeffs as a list

    for i in range(0, h-blkSize+1, blkSize):
        for j in range(0, w-blkSize+1, blkSize):
            blk = gray[i:i+blkSize, j:j+blkSize]
            dblk = dct2(blk)
            #dblk = cv2.bitwise_and(dblk, dblk, mask=mask) #can use this statement with the commented statement above - that pushes values to 255 and 0

            if intQ == True:
                dblk = intQuantizeDctBlk(dblk)

            dblk = np.where(mask<K, dblk, 0) #easier, one line

            for i_ord, j_ord in ordIdx:
                allRetainedCoeffs.append(dblk[i_ord,j_ord])

            idblk = idct2(dblk)
            outImg[i:i+blkSize, j:j+blkSize] = idblk
            dctFrame[i:i+blkSize, j:j+blkSize] = dblk

    allRetainedCoeffs = np.array(allRetainedCoeffs).reshape(1,-1)

    return allRetainedCoeffs


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
                out, _ = flipAndRotateAlign(gray)
                allretcoeffs = dctBlockWise(out, 8, 48)
                flat_gray = allretcoeffs.reshape(-1,)
                for value in flat_gray:
                        data = struct.pack('<h', value)
                        audioObj.writeframesraw(data)


if __name__ == "__main__":
    all_vid_path = sys.argv[1]
    for vid_path in os.listdir(all_vid_path):
            convert_video2audio1(all_vid_path+vid_path)
    
