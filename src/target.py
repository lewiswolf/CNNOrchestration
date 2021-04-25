import os, random
import numpy as np						# maths
import soundfile as sf					# audio i/o
import librosa							# audio tools

# import global vars
from settings import PATH_TO_DATASET, SAMPLES_PER_TARGET, SAMPLE_RATE

class Target:
	'''
	A class for combining instrumental samples into one big sample,
	complete with metadata calibration.
	'''

	def __init__(self):
		self.instruments = []
		self.labels = []
		self.sampleFilepaths = []
		self.renderedFilepath = ''

	def combine_for_training(self, index):
		output = np.zeros(SAMPLE_RATE * 4)
		tmpList = []
		longest_sample = 0

		# populate tmp list and find longest sample
		for i in range(SAMPLES_PER_TARGET):
			audio, sr = sf.read(os.path.join(PATH_TO_DATASET, self.sampleFilepaths[i]))
			if (sr != SAMPLE_RATE):
				librosa.resample(audio, sr, SAMPLE_RATE)
			tmpList += [audio]
			if (len(tmpList[i]) > longest_sample):
				longest_sample = len(tmpList[i])

		# pad and combine samples into a numpy array
		tmpNP = np.zeros((longest_sample))
		for i in range(SAMPLES_PER_TARGET):
			prefix = random.randint(0, longest_sample - len(tmpList[i]))
			for j in range(len(tmpList[i])):
				tmpNP[j + prefix] += tmpList[i][j]

		# reduce amplitude and trim output
		tmpNP /= SAMPLES_PER_TARGET
		tmpNP = tmpNP * (1.0 / np.max(tmpNP))
		trim = (longest_sample - output.shape[0]) // 2
		for i in range(output.shape[0]):
			output[i] = tmpNP[i + trim]

		# set metadata and export target
		outputPath = f'targets/target_{index}.wav'
		self.renderedFilepath = outputPath
		sf.write(os.path.join(os.getcwd(), outputPath), output, SAMPLE_RATE)
	
	def getMetadata(self):
		return {
			'labels': self.labels,
			'filepath': self.renderedFilepath,
		}