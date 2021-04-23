import json, os

# GLOBAL VARS
PATH_TO_DATASET = os.path.join(os.getcwd(), 'TinySOL')
SAMPLES_PER_TARGET = 10
NUM_OF_TARGETS = 300
SAMPLE_RATE = 44100

LEARNING_RATE = 0.5
BATCH_SIZE = 4
NUM_OF_EPOCHS = 100

PATH_TO_TRAINED_MODEL = ''

def export_settings_to_json(export_path):
	settings = {
		'SAMPLES_PER_TARGET': SAMPLES_PER_TARGET,
		'NUM_OF_TARGETS': NUM_OF_TARGETS,
		'SAMPLE_RATE': SAMPLE_RATE,
		'LEARNING_RATE': LEARNING_RATE,
		'BATCH_SIZE': BATCH_SIZE,
		'NUM_OF_EPOCHS': NUM_OF_EPOCHS
	}

	with open(f'{export_path}.json', 'w') as json_file:
		json.dump(settings, json_file)	