import os
import sys
import subprocess

#sys.argv[1] is path to WAVS
#sys.argv[2] is database 
#sys.argv[3] is result prefix
correct = 0
total = 0

key = sys.argv[3]

with open("results_"+key,"w") as f:
    for wav in os.listdir(sys.argv[1]):
        try:
            proc = subprocess.check_output('python3 audfprint.py match --min-count '+sys.argv[4]+' --dbase '+sys.argv[2]+' '+sys.argv[1]+wav,shell=True)
#            proc = subprocess.check_output('python3 audfprint.py match --dbase '+sys.argv[2]+' '+sys.argv[1]+wav,shell=True)
            proc = proc.decode('utf-8')
            proc = proc.split("\n")[2]
     #       print(proc)
            if proc == "NOMATCH":
                print(wav+",0")
                f.write(wav+",0\n")
            else:
                ps = proc.split(",")
                wav1 = ps[1].strip("\r\t\n").split("/")[-1]
                wav2 = ps[2].strip("\r\t\n").split("/")[-1]
             #   print(ps[1].strip("\r\t\n"))
              #  print(ps[2].strip("\r\t\n"))
                if wav1 == wav2:
                    correct+=1
                    print(wav+",1")
                    f.write(wav+",1\n")
                else:
                  #  print(wav1+","+wav2+",-1")
                   # f.write(wav1+","+wav2+",-1")
                    print(wav+",0")
                    f.write(wav+",0\n")
            total+=1
        except:
            pass
    print("Accuracy for "+sys.argv[2]+" seconds input: "+str(correct/total))
    f.write("Accuracy for "+sys.argv[2]+" seconds input: "+str(correct/total)+"\n")



    

