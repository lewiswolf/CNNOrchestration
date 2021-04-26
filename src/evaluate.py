import json, os, random, sys
import torch
import numpy as np
import soundfile as sf

from settings import PATH_TO_DATASET, PATH_TO_MODEL
from generateTargets import import_original_metadata
from nn import NeuralNet, import_to_mel

def load_model():
	'''
	Loads a pre-trained model and settings used to generate it.
	'''
	try:
		with open(f'{PATH_TO_MODEL}.json', 'r') as json_file:
			settings = json.load(json_file)
		model = NeuralNet(len(settings['all_labels']))
		model.load_state_dict(torch.load(f'{PATH_TO_MODEL}.pth', map_location=torch.device('cpu')))
	except:
		print('Could not locate a trained model.')
		sys.exit()
	model.eval()
	return model, settings

def orchestrate_target(eval_model, settings, custom_target):
	'''
	Orchestrate a custom target using a pretrained model.
	input:
		trained model (model.eval())
		a dict of settings used to train the model
	output:
		a rendered audio file in \out
	'''
	# get original metadata
	original_metadata = import_original_metadata()
	print('Orchestrating target... 🎻')
	# set device
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	# load model
	eval_model = eval_model.to(device)
	# load target
	target = import_to_mel(custom_target, settings['SAMPLE_RATE'])
	target = torch.unsqueeze(target, 0)
	target = target.to(device)

	with torch.no_grad():
		# find the most likely labels
		y_predicted = eval_model(target).cpu()
		y_predicted = y_predicted.numpy().flatten()
		idx_predicted = np.argsort(y_predicted)[0 - settings['SAMPLES_PER_TARGET'] :]
		
		tmp = []
		for i in idx_predicted:
			# configure labels
			label = settings['all_labels'][i]
			dynamic = 'pp' if y_predicted[i] < 0.33 else 'ff' if y_predicted[i] > 0.66 else 'mf'

			# find the sample that matches labels
			metadata_subset = original_metadata[original_metadata['Instrument (abbr.)'] == label[0]]
			metadata_subset = metadata_subset[metadata_subset['Pitch'] == label[1]]
			metadata_subset = metadata_subset[metadata_subset['Dynamics'] == dynamic]
			metadata_subset.reset_index(drop=True, inplace=True)
			if (len(metadata_subset) > 0):
				sample_path = metadata_subset['Path'][random.randint(0, len(metadata_subset) - 1)]

			# import sample
			waveform = sf.read(os.path.join(PATH_TO_DATASET, sample_path))[0]
			tmp += [waveform]

		# combine and normalise
		longest_sample = len(max(tmp, key=len))
		out = np.zeros(longest_sample)
		for sample in tmp:
			for i in range(len(sample)):
				out[i] += sample[i]
		out = out * (1.0 / np.max(out)) # normalise

		# save output
		filename = os.path.basename(custom_target).removesuffix('.wav')
		sf.write(os.path.join(os.getcwd(), f'out/{filename}.wav'), out, settings['SAMPLE_RATE'])
		print('Target Orchestrated 🎉')