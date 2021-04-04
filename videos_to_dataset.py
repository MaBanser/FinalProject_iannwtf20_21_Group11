# Libraries and Settings
import os
import cv2
import moviepy.editor as mp
import tensorflow as tf
import numpy as np
import pickle5 as pickle #pip install pickle5
import bz2
from scipy import signal

aspect_ratio = 16/9
height = 96
width  = int(height*aspect_ratio)

clip_len = 3
sample_rate = 16000
fps = 3

audio_step_size = (1.0/fps)*sample_rate
video_step_size = (1.0/fps)


def getFrame(video, sec):
	video.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
	hasFrames,image = video.read()
	return image

# Pickle a file and then compress it into a file with extension
def compressed_pickle(title, data):
	with bz2.BZ2File(title + ".pbz2", "w") as f:
		pickle.dump(data, f)


# iterate through files
dataset_images = []
dataset_audio = []

path = "videos/"
for subdir, dirs, files in os.walk(path):
	for file in files:
		filepath = f"{subdir}{os.sep}{file}"

		# extract audio and video from file (we take only one channel)
		try:
			audio = mp.AudioFileClip(filepath)
		except OSError:
			print("encountered non video file")
			continue

		audio_array = audio.to_soundarray()[:,0]

		# downsample audio
		audio_array = signal.resample(audio_array, int(len(audio_array) * (sample_rate/44100)))

		video = cv2.VideoCapture(filepath)

		# split into clips
		start, stop, step = 0, len(audio_array), clip_len*sample_rate
		for i in range(start, stop-step, step):
			frames_of_clip = []
			audio_of_clip = []

			clip_audio = audio_array[i:i+step]
			audio_of_clip.append(clip_audio)

			for j in range(fps*clip_len):
				image = getFrame(video, (i/sample_rate)+(j+0.5)*video_step_size)
				image = cv2.resize(image, dsize=(width, height), interpolation=cv2.INTER_CUBIC)
				image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
				frames_of_clip.append(image)

			dataset_images.append(frames_of_clip)
			dataset_audio.append(audio_of_clip)

		print(f"{file} written to dataset")

# convert to numpy array
dataset_images = np.array(dataset_images)
dataset_audio = np.array(dataset_audio)


# serialize dataset
compressed_pickle("dataset_images", dataset_images)
compressed_pickle("dataset_audio", dataset_audio)

print("Datasets serialized")
