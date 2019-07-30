# processing the raw data of the video datasets (Something-something and jester)
# generate the meta files:
#   category.txt:               the list of categories.
#   train_videofolder.txt:      each row contains [videoname num_frames classIDX]
#   val_videofolder.txt:        same as above
#
# Bolei Zhou, Dec.2 2017
#
#
import os
import pdb

import json

dataroot = '/scr1/workspace/dataset/sth-sth'
labelfile = dataroot+'/captions/something-something-v2-labels.json'
trainfile = dataroot+'/captions/something-something-v2-train.json'
valfile = dataroot+'/captions/something-something-v2-validation.json'
testfile = dataroot+'/captions/something-something-v2-test.json'

extract_frames = dataroot+'/extract_frames'
dict_categories = json.load(open(labelfile))

files_input = [valfile,trainfile]
files_output = ['val_videofolder.txt','train_videofolder.txt']
for (filename_input, filename_output) in zip(files_input, files_output):
    with open(filename_input) as f:
        lines = json.load(f)
    folders = []
    idx_categories = []

    for line in lines:
        folders.append(int(line['id']))
        ground_truth = line['template'].replace('[','')
        ground_truth = ground_truth.replace(']','')
        idx_categories.append(dict_categories[ground_truth])
    output = []

    for i in range(len(folders)):
        curFolder = str(folders[i])
        curIDX = int(idx_categories[i])
        # counting the number of frames in each video folders
        dir_files = os.listdir(os.path.join(extract_frames, curFolder))
        output.append('%s %d %d'%(curFolder, len(dir_files), curIDX))
        print('%d/%d'%(i, len(folders)))
    with open(filename_output,'w') as f:
        f.write('\n'.join(output))
