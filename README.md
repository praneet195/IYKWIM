# If You Know What I've Seen
## An Audio Landmark Detection based approach for Video Fingerprinting

This repository contains the code of our audio landmark detection based video fingerprinting technique that is simple yet powerful
</br>

Package Requirements:
```
numpy
openCV
docopt
scipy
joblib
psutil
audfprint
```

## Usage:
### UCF-101 Database Setup
```
-> Download the UCF-101 database from www.crcv.ucf.edu/data/UCF101.php
-> Keep only the videos listed in vids_greater_15s.txt
  -> These will be used to create the database
-> To create queries that have the first "N" frames use create_queries.py
  -> python3 create_queries.py <Path_to_UCF_Dataset>
```
### Convert Videos to Pseudo-Audio Files
Use video2audio.py to convert video files to pseudo-audio wav files </br>
Usage: python3 video2audio_`<distortion>`.py [Path to Frames] [Output Wav Folder] [Distortion_Value]

```
--input_pth           Path to Video Frames
--output_pth          Output folder where Wavs are created
```

##### Example Runs:
```
Create Wavs from Undistorted frames:
-> python3 video2audio.py  <Path To DB/Query Frames> <Output Wav Folder>  
  -> This resizes all the frames to 128x128. This can be easily changed to any other scale required.

If Disortions need to be added, follow the steps below:
Resizing:
-> python3 video2audio_resize.py  <Path To Query Frames> <Output Wav Folder>  <widthxheight>

Drop:
-> Create a copy of the Query Folder 
-> Use drop_frames_randomly.py to create queries with "n" frames dropped randomly per second on the copied version of the query folder
  -> python3 drop_frames_randomly.py <Path_To_Query_Frames_Copy> <Frames_Dropped_Per_Second>
-> python3 video2audio.py  <Path_To_Query_Frames_Copy> <Output Wav Folder> 

Rotation:
-> python3 video2audio_rotation.py  <Path To Query Frames> <Output Wav Folder>  <90,180>

Flip:
-> python3 video2audio_flip.py  <Path To Query Frames> <Output Wav Folder>  <hor,ver>

Watermark:
-> python3 video2audio_watermark.py  <Path To Query Frames> <Output Wav Folder>  <TL,TR,BL,BR> <watermark_image>

Compression:
-> First run x265 compression on the query frames. QPs used for compression [10,20,30,40,50]
  -> All-Intra: python3 qp_encode_frames.py <Path To Query Frames> <Path to Save Compressed Frames>
  -> Inter: python3 qp_encode_frames_inter.py <Path To Query Frames> <Path to Save Compressed Frames>

```
### Test Fingerprinting
```
-> Setup audfprint using the steps listed @ https://github.com/dpwe/audfprint

-> Create database as follows:
  -> python3 audfprint.py new --dbase <database_name>.pklz <Path to Wavs>/*.wav
    -> Example: python3 audfprint.py new --dbase ucf_database.pklz UCF_Full_Wavs/*.wav
    
-> Run the command to get fingerprinting accuracy:
  -> python3 get_results_generic.py <Path to Query Wavs> <Database Name> <Result File Name> 
   -> Example: python3 get_results_generic.py UCF_Query_Wavs/ ucf_database.pklz ucf_baseline.txt 

```

## Mitigitation Strategies

### Scale Pyramid to Improve Fingerprinting on High Resolution Videos (720p):
```
-> Run video2audio.py at three different scales for both the database and query videos i.e 128x128, 256x256 and 384x384. Also, adjust the sample rate to the frame rate of the videos
-> Create the database at all three scales. Also, these databases can be merged using the "merge" command in audfprint.
-> Test the query videos at each scale and merge results
```

### Rotation and Flip Invariance:

```
-> Run video2audio_plus.py/video2audio_plusplus.py on both query segments and database videos (same procedure as running video2audio.py stated earlier)
-> Create the database using the Rotation and Flip Invariant database videos
-> Test the new Rotation and Flip Invariant Query segments on this database
```







