import json, os, random, sys
import click						# cli
import pandas as pd					# databases
import numpy as np					# maths
import soundfile as sf				# audio i/o
import librosa						# audio tools

# import global vars
from settings import PATH_TO_DATASET, NUM_OF_TARGETS, SAMPLES_PER_TARGET, SAMPLE_RATE

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
				audio = librosa.resample(audio, sr, SAMPLE_RATE)
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

def import_original_metadata():
	'''
	Import the original metadata
	returns a pandas dataframe
	'''
	try:
		metadata = pd.read_csv(
			os.path.join(PATH_TO_DATASET, 'TinySOL_metadata.csv'),
			engine='python'
		)
	except:
		print('TinySOL metadata not found.')
		sys.exit()
	return metadata

def generate_targets():
	'''
	Generates targets for source seperation based on model settings.
	Returns metadata for the targets, and saves the rendered wav 
	files and the metadata.json in the /targets folder
	'''
	print('Generating source seperation targets... 🎯')

	# get dataset metadata
	datasetMetadata = import_original_metadata()

	# initialise targets metadata
	targetsJSON = {
		"SAMPLES_PER_TARGET": SAMPLES_PER_TARGET,
		"NUM_OF_TARGETS": NUM_OF_TARGETS,
		"SAMPLE_RATE": SAMPLE_RATE,
		"all_labels": [],
		"targets": []
	}

	# clear folder
	targetsFolder = os.path.join(os.getcwd(), 'targets')
	for file in os.listdir(targetsFolder):
		if (file != '.gitignore'):
			os.remove(os.path.join(targetsFolder, file))

	# create list of labels
	labels = []
	for i in range(len(datasetMetadata)):
		tmp = [datasetMetadata['Instrument (abbr.)'][i], datasetMetadata['Pitch'][i]]
		if (not tmp in labels):
			labels += [tmp]
	targetsJSON['all_labels'] = labels

	with click.progressbar(length = NUM_OF_TARGETS) as bar:
		for i in range(NUM_OF_TARGETS):
			target = Target()
			for j in range(SAMPLES_PER_TARGET):
				# generate a random datasample
				index = random.randint(0, len(datasetMetadata) - 1)
				# regenerate if there are more than 3 instances of this instrument
				while(target.instruments.count(datasetMetadata['Instrument (abbr.)'][index]) >= 3):
					index = random.randint(0, len(datasetMetadata) - 1)
				# append metadata to class
				target.instruments += [datasetMetadata['Instrument (abbr.)'][index]]
				target.sampleFilepaths += [datasetMetadata['Path'][index]]
				target.labels += [labels.index([datasetMetadata['Instrument (abbr.)'][index], datasetMetadata['Pitch'][index]])]
				
			# combine samples
			target.combine_for_training(i)

			# append metadata to output json and update progressbar
			targetsJSON['targets'] += [target.getMetadata()]
			bar.update(1)

	# export metadata as json
	with open(os.path.join(os.getcwd(), 'targets/metadata.json'), 'w') as json_file:
		json.dump(targetsJSON, json_file)

	return targetsJSON