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
from evaluate import load_model, orchestrate_target

# set command line flags
@click.command()
@click.option('--train', '-t', is_flag = True, help = 'Train a new model.')
@click.option('--generate', '-g', is_flag = True, help = 'Generate targets before training.')
@click.option('--orchestrate', '-o', is_flag = True, help = 'Orchestrate a target sound.')
def main(train, generate, orchestrate):
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
		final_model, accuracy = train_model(train_dataset, test_dataset, len(targetsJSON['all_labels']))

		# save model and settings
		export_path = os.path.join(os.getcwd(), f'models/model_{datetime.now().strftime("%d%m%y_%H%M")}')
		torch.save(final_model.state_dict(), f'{export_path}.pth')
		train_settings = export_settings()
		train_settings['Final Accuracy'] = accuracy
		train_settings['all_labels'] = targetsJSON['all_labels']
		with open(f'{export_path}.json', 'w') as json_file:
			json.dump(train_settings, json_file)
		print('Model saved! 📁')

	# orchestrate a user-defined sample
	if (orchestrate):
		try:
			# use the model just trained
			eval_model = final_model.eval()
			eval_settings = train_settings
		except:
			# or load an existing one
			eval_model, eval_settings = load_model()
		
		# get filepath and evaluate
		custom_target = click.prompt('What is the filepath to the target sound?', type=str)[1:-1]
		orchestrate_target(eval_model, eval_settings, custom_target)

	print('')

if __name__ == '__main__':
    main()