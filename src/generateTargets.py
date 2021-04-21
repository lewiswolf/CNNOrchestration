import json, os, random, sys
import click							# cli
import pandas as pd						# databases

# import global vars
from settings import PATH_TO_DATASET, NUM_OF_TARGETS, SAMPLES_PER_TARGET, SAMPLE_RATE
from target import Target

def generate_targets():
	'''
	Generates targets for source seperation based on model settings.
	Returns metadata for the targets, and saves the rendered wav 
	files and the metadata.json in the /targets folder
	'''

	print('Generating source seperation targets... 🎯')

	# get dataset metadata
	try:
		datasetMetadata = pd.read_csv(
			os.path.join(PATH_TO_DATASET, 'TinySOL_metadata.csv'),
			engine='python'
		)
	except:
		print('TinySOL metadata not found.')
		sys.exit()

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