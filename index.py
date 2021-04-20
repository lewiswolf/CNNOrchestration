import json, os, sys
import click					# cli

# add /src to sys.path
sys.path.insert(1, os.path.join(os.getcwd(), 'src'))
import generateTargets

# GLOBAL VARS
PATH_TO_DATASET = os.path.join(os.getcwd(), 'TinySOL')
SAMPLES_PER_TARGET = 10
NUM_OF_TARGETS = 300
SAMPLE_RATE = 44100

# set command line flags
@click.command()
@click.option('--generate', '-g', is_flag = True, help = 'Generate targets before training.')
def main(generate):
	# generate targets and store metadata
	targets = None

	if (generate):
		targets = generateTargets.generateTargets()
	else:
		try:
			# load targets metadata
			targets = json.load(open(os.path.join(os.getcwd(), 'targets/metadata.json'), 'r'))
		except:
			targets = generateTargets.generateTargets()
		# if the project settings and data settings don't align, regenerate
		if (targets['SAMPLES_PER_TARGET'] != SAMPLES_PER_TARGET or targets['NUM_OF_TARGETS'] != NUM_OF_TARGETS or targets['SAMPLE_RATE'] != SAMPLE_RATE):
			targets = generateTargets.generateTargets()

if __name__ == '__main__':
    main()