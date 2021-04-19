import sys, os
import click

# add /src to sys.path
sys.path.insert(1, os.path.join(os.getcwd(), 'src'))
import generateTargets

# GLOBAL VARS
SAMPLES_PER_TARGET = 2
NUM_OF_TARGETS = 200
PATH_TO_DATASET = os.path.join(os.getcwd(), 'TinySOL')

targets = None

# set command line flags
@click.command()
@click.option('--generate', '-g', is_flag = True, help = 'Generate targets before training.')
def main(generate):
	# options
	GENERATE_TARGETS = generate

	# check if targets folder is populated
	if (len(os.listdir('./targets')) - 1 != NUM_OF_TARGETS or GENERATE_TARGETS):
		# check if dataset exists
		if (not os.path.isdir(PATH_TO_DATASET)):
			print('Dataset not found.')
			sys.exit()
		# generate targets
		print('Generating source seperation targets... 🎯')
		targets = generateTargets.generateTargets(PATH_TO_DATASET, SAMPLES_PER_TARGET)
	else:
		targets = None # load targets metadata


if __name__ == '__main__':
    main()

	# import pandas as pd		# databases
	# import torch				# pytorch
	# import torchaudio			# handling audio

	# # import metadata
	# metadata = pd.read_csv('./TinySOL_2020/TinySOL_metadata.csv', engine='python')

	# # for i in range(len(metadata)):
	# waveform, sampleRate = torchaudio.load('./TinySOL_2020/TinySOL/Brass/BTb/ordinario/BTb-ord-A#1-ff.wav')
	# # waveform, sampleRate = torchaudio.load_wav(f'./TinySOL_2020/{metadata['Path'][i]}')
	# spectrogram = torchaudio.transforms.MelSpectrogram(
	# 	sample_rate = sampleRate,
	# 	n_mels = 128,
	# 	hop_length = 2048,
	# 	pad = int(((44100 * 4) - waveform.size()[1]) / 2)
	# )(waveform[0])