import os, sys
# import itertools, math
import pandas as pd		# databases
import numpy as np		# maths

# class Sample:
# 	def __init__(self):

def importMetadata(PATH_TO_DATASET):
	pathToMetadata = os.path.join(PATH_TO_DATASET, 'TinySOL_metadata.csv')
	try:
		metadata = pd.read_csv(pathToMetadata, engine='python')
	except:
		print('Metadata not found.')
		sys.exit()
	return metadata

def generateTargets(PATH_TO_DATASET, SAMPLES_PER_TARGET):
	metadata = importMetadata(PATH_TO_DATASET)
	# subsets = []
	# for i in range(math.floor(len(metadata) / SAMPLES_PER_TARGET)):
	# 	tmp = []
	# 	for j in range(SAMPLES_PER_TARGET):
	# 		tmp.append(metadata['Path'][i * SAMPLES_PER_TARGET + j])
	# 	subsets += [tmp]