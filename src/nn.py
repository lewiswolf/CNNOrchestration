import os
import click 				# cli
import torch				# pytorch
import torchaudio			# handling audio

from settings import NUM_OF_EPOCHS, SAMPLE_RATE

def train_model(targetsMetadata):
	'''
	A CNN for classifying source seperation.
	params: metadata array containing objects corresponding to each target.
		target = {
			'labels': array of labels for each sample [[Flute, G3, pp], [Violin, A5, ff], ... ]
			'filepath': filepath to target soundfile
		}
	'''

	print('Training neural network... 🧠')

	# configure hardware
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	# with click.progressbar(length = NUM_OF_EPOCHS) as bar:
		# for epoch in range(NUM_OF_EPOCHS):


			# waveform = torchaudio.load(os.path.join(os.getcwd(), f'targets/target_{0}.wav'))[0] # add the correct iterator later
			# print(waveform)



			# spectrogram = torchaudio.transforms.MelSpectrogram(
			# 	sample_rate = SAMPLE_RATE,
			# 	n_mels = 128,
			# 	hop_length = 2048,
			# 	pad = int(((44100 * 4) - waveform.size()[1]) / 2)
			# )(waveform)

			# bar.update(1)