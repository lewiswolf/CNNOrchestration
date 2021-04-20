import json, os, random, sys
import click							# cli
import pandas as pd						# databases
import numpy as np						# maths
import soundfile as sf					# audio i/o
import librosa							# audio tools

# import global vars
from index import PATH_TO_DATASET, NUM_OF_TARGETS, SAMPLES_PER_TARGET, SAMPLE_RATE

class Sample:
	def __init__(self):
		self.instruments = []
		self.labels = []
		self.sampleFilepaths = []
		self.renderedFilepath = ''

	def combineSamples(self, index):
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
		trim = (longest_sample - output.shape[0]) // 2
		for i in range(output.shape[0]):
			output[i] = tmpNP[i + trim]

		# set metadata and export target
		outputPath = os.path.join(os.getcwd(), f'targets/target_{index}.wav')
		self.renderedFilepath = outputPath
		sf.write(outputPath, output, SAMPLE_RATE)
	
	def getMetadata(self):
		return {
			'labels': self.labels,
			'filepath': self.renderedFilepath,
		}


def generateTargets():
	print('Generating source seperation targets... 🎯')

	# get dataset metadata
	try:
		datasetMetadata = pd.read_csv(
			os.path.join(PATH_TO_DATASET, 'TinySOL_metadata.csv'),
			engine='python'
		)
	except:
		print('Metadata not found.')
		sys.exit()

	# initialise targets metadata
	targetsMetadata = {
		"SAMPLES_PER_TARGET": SAMPLES_PER_TARGET,
		"NUM_OF_TARGETS": NUM_OF_TARGETS,
		"SAMPLE_RATE": SAMPLE_RATE,
		"targets": []
	}

	with click.progressbar(length = NUM_OF_TARGETS) as bar:
		for i in range(NUM_OF_TARGETS):
			sample = Sample()
			for j in range(SAMPLES_PER_TARGET):
				# generate a random datasample
				index = random.randint(0, len(datasetMetadata) - 1)
				# regenerate if there are more than 3 instances of this instrument
				while(sample.instruments.count(datasetMetadata['Instrument (abbr.)'][index]) >= 3):
					index = random.randint(0, len(datasetMetadata) - 1)
				# append metadata to class
				sample.instruments += [datasetMetadata['Instrument (abbr.)'][index]]
				sample.sampleFilepaths += [datasetMetadata['Path'][index]]
				sample.labels += [[
					datasetMetadata['Instrument (in full)'][index], 
					datasetMetadata['Pitch'][index], 
					datasetMetadata['Dynamics'][index]
				]]
				
			# combine samples
			sample.combineSamples(i)

			# append metadata to output json and update progressbar
			targetsMetadata['targets'] += [sample.getMetadata()]
			bar.update(1)

	# export metadata as json
	with open(os.path.join(os.getcwd(), 'targets/metadata.json'), 'w') as json_file:
		json.dump(targetsMetadata, json_file)

	return targetsMetadata