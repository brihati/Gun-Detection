import os
from constants import DATA_PATH,DATA_FILE_NAME,GUN_TYPE
import csv
import random

onlyDirs=[d for d in os.listdir(DATA_PATH)]
data=[]
for dir in onlyDirs:
    path=DATA_PATH
    path+='\\'
    path+=dir
    onlyfiles = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    if(dir==GUN_TYPE[0]):
        label=0
    elif(dir==GUN_TYPE[1]):
        label=1
    elif (dir==GUN_TYPE[2]):
        label = 2
    elif (dir==GUN_TYPE[3]):
        label = 3
    else:
        label = 4

    for fileName in onlyfiles:
        row=[]
        row.append(path+"\\"+fileName)
        row.append(label)
        data.append(row)
with open(DATA_FILE_NAME,"w",newline='') as csvfile:
    csv_writer=csv.writer(csvfile,delimiter=',')
    for row in data:
        csv_writer.writerow([row[0],row[1]])

### Shuffle the lines
lines=open(DATA_FILE_NAME).readlines()
random.shuffle(lines)
open(DATA_FILE_NAME, 'w',newline='').writelines(lines)
