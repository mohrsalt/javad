import os
import numpy as np
from javad.src.javad.extras.basic import get_speech_intervals
import pandas as pd
import datasets
import librosa
import json
from tqdm import tqdm

with open("/home/users/ntu/bhargavi/scratch/test_dataset_top20/metadata_test_A.json", "r", encoding="utf-8") as f:
    data = json.load(f)


columns=["AudioPath","Reference","Hypotheses","Logits","Length"]
preds=pd.DataFrame(columns=columns)

trained_model="/home/users/ntu/bhargavi/balanced_test.pt"

total=0
root="/home/users/ntu/bhargavi/scratch/test_dataset_top20/dev/"
for file in tqdm(data):
	file_loc=root+file
	waveform_32k, _ = librosa.load(file_loc, sr=32000, mono=True)
	logits,predicted_intervals = get_speech_intervals(file_loc,trained_model)
	ground_truth = data[file]["intervals"]["speech"]
	total=total+len(waveform_32k)
	preds.loc[len(preds)] = [file_loc, ground_truth,predicted_intervals,logits,len(waveform_32k)/32000]

print(total/(32000*3600))
preds.to_csv("/home/users/ntu/bhargavi/JavadPreds_top20.csv")


