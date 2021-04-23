import json, os, sys
from datetime import datetime
import click					# cli
import torch					# pytorch

# add /src to sys.path
sys.path.insert(1, os.path.join(os.getcwd(), 'src'))
from settings import NUM_OF_TARGETS, SAMPLES_PER_TARGET, SAMPLE_RATE
from settings import export_settings_to_json
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
		# format to torch dataset class
		dataset = TargetsDataset(targetsJSON['targets'], len(targetsJSON['all_labels']))
		print('Dataset loaded! 🗄')

		# train the model
		final_model = train_model(dataset, targetsJSON['all_labels'])

		# save model and settings
		export_path = os.path.join(os.getcwd(), f'models/model_{datetime.now().strftime("%d%m%y_%H%M")}')
		torch.save(final_model.state_dict(), f'{export_path}.pth')
		export_settings_to_json(export_path)

	print('')

if __name__ == '__main__':
    main()