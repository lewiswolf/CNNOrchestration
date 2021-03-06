import os

# GLOBAL VARS
PATH_TO_DATASET = os.path.join(os.getcwd(), 'TinySOL')
SAMPLES_PER_TARGET = 10
NUM_OF_TARGETS = 1000
SAMPLE_RATE = 44100

LEARNING_RATE = 0.5
BATCH_SIZE = 4
NUM_OF_EPOCHS = 50

PATH_TO_MODEL = os.path.join(os.getcwd(), 'example/models/model_260421_1952') # don't include extension

def export_settings():
	settings = {
		'SAMPLES_PER_TARGET': SAMPLES_PER_TARGET,
		'NUM_OF_TARGETS': NUM_OF_TARGETS,
		'SAMPLE_RATE': SAMPLE_RATE,
		'LEARNING_RATE': LEARNING_RATE,
		'BATCH_SIZE': BATCH_SIZE,
		'NUM_OF_EPOCHS': NUM_OF_EPOCHS
	}
	return settings