import json, os, sys
import click					# cli

# add /src to sys.path
sys.path.insert(1, os.path.join(os.getcwd(), 'src'))
from settings import NUM_OF_TARGETS, SAMPLES_PER_TARGET, SAMPLE_RATE
from generateTargets import generateTargets
from nn import train_model

# set command line flags
@click.command()
@click.option('--generate', '-g', is_flag = True, help = 'Generate targets before training.')
def main(generate):
	# generate targets and store metadata
	targets = None
	if (generate):
		targets = generateTargets()
	else:
		try:
			# load targets metadata
			targets = json.load(open(os.path.join(os.getcwd(), 'targets/metadata.json'), 'r'))
		except:
			targets = generateTargets()
		# if the project settings and data settings don't align, regenerate
		# this only happens if the import works
		if (targets['SAMPLES_PER_TARGET'] != SAMPLES_PER_TARGET or targets['NUM_OF_TARGETS'] != NUM_OF_TARGETS or targets['SAMPLE_RATE'] != SAMPLE_RATE):
			targets = generateTargets()
	print('Dataset loaded! 🗄')

	# train the model
	train_model(targets['targets'])

if __name__ == '__main__':
    main()