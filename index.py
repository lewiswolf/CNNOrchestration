import json, os, sys
from datetime import datetime
import click					# cli
import torch					# pytorch

# add /src to sys.path
sys.path.insert(1, os.path.join(os.getcwd(), 'src'))
from settings import NUM_OF_TARGETS, SAMPLES_PER_TARGET, SAMPLE_RATE
from settings import export_settings
from generateTargets import generate_targets
from nn import TargetsDataset, train_model

# set command line flags
@click.command()
@click.option('--train', '-t', is_flag = True, help = 'Generate targets before training.')
@click.option('--generate', '-g', is_flag = True, help = 'Generate targets before training.')
def main(train, generate):
	print('')

	if (train):
		# generate targets and store metadata
		if (generate):
			targetsJSON = generate_targets()
		else:
			try:
				# load targets metadata
				targetsJSON = json.load(open(os.path.join(os.getcwd(), 'targets/metadata.json'), 'r'))
			except:
				targetsJSON = generate_targets()
			# if the project settings and data settings don't align, regenerate
			# this only happens if the import works
			if (
				targetsJSON['SAMPLES_PER_TARGET'] != SAMPLES_PER_TARGET or 
				targetsJSON['NUM_OF_TARGETS'] < NUM_OF_TARGETS or 
				targetsJSON['SAMPLE_RATE'] != SAMPLE_RATE
			):
				targetsJSON = generate_targets()
			# make targets metadata the same length as NUM_OF_TARGETS
			if (targetsJSON['NUM_OF_TARGETS'] != NUM_OF_TARGETS):
				targetsJSON['targets'] = targetsJSON['targets'][: NUM_OF_TARGETS]

		# format to torch datasets
		print('Preprocessing dataset... 📝')
		size_of_training_set = round(NUM_OF_TARGETS * 0.7)
		train_dataset = TargetsDataset(targetsJSON['targets'][: size_of_training_set], len(targetsJSON['all_labels']))
		test_dataset = TargetsDataset(targetsJSON['targets'][size_of_training_set :], len(targetsJSON['all_labels']))
		print('Dataset loaded! 🗄')

		# train the model
		final_model, accuracy = train_model(train_dataset, test_dataset, targetsJSON['all_labels'])

		# save model and settings
		export_path = os.path.join(os.getcwd(), f'models/model_{datetime.now().strftime("%d%m%y_%H%M")}')
		torch.save(final_model.state_dict(), f'{export_path}.pth')
		settings = export_settings()
		settings['Final Accuracy'] = accuracy
		with open(f'{export_path}.json', 'w') as json_file:
			json.dump(settings, json_file)	
		print('Model saved! 📁')

	print('')

if __name__ == '__main__':
    main()