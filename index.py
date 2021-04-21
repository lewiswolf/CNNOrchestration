import json, os, sys
import click					# cli

# add /src to sys.path
sys.path.insert(1, os.path.join(os.getcwd(), 'src'))
from settings import NUM_OF_TARGETS, SAMPLES_PER_TARGET, SAMPLE_RATE
from generateTargets import generate_targets
from nn import TargetsDataset, train_model

# set command line flags
@click.command()
@click.option('--generate', '-g', is_flag = True, help = 'Generate targets before training.')
def main(generate):
	print('')

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
			targetsJSON['NUM_OF_TARGETS'] != NUM_OF_TARGETS or 
			targetsJSON['SAMPLE_RATE'] != SAMPLE_RATE
		):
			targetsJSON = generate_targets()
	# format to torch dataset class
	dataset = TargetsDataset(targetsJSON['targets'], len(targetsJSON['all_labels']))
	print('Dataset loaded! 🗄')

	# train the model
	train_model(dataset, targetsJSON['all_labels'])

	print('')

if __name__ == '__main__':
    main()