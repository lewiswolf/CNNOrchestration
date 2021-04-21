import os
import click 				# cli
import torch				# pytorch
import torchaudio			# handling audio
import numpy as np			# maths
import pandas as pd			# databases
import matplotlib.pyplot as plt

from settings import NUM_OF_EPOCHS, SAMPLE_RATE

class TargetsDataset(torch.utils.data.Dataset):
	'''
	The dataset
		self.X is a list of mel spectrograms
		self.Y is a list of lists containing the labels.
	'''

	def __init__(self, targets):
		# transform json into lists and preprocess
		self.X = self.preprocess_x(pd.DataFrame(targets)['filepath'].values)
		self.Y = pd.DataFrame(targets)['labels'].values
		self.n_samples = self.Y.shape[0]
		print(self.X[0].shape)

	# helper methods
	def __getitem__(self, index):
		return self.X[index], self.Y[index]
	def __len__():
		return self.n_samples

	# generate spectrograms from rendered targets
	def preprocess_x(self, filepaths):
		print('Preprocessing dataset... 📝')
		X = []
		for x in filepaths:
			waveform = torchaudio.load(os.path.join(os.getcwd(), x))[0]
			spectrogram = torchaudio.transforms.MelSpectrogram(
				sample_rate = SAMPLE_RATE,
				n_mels = 128,
				n_fft = 2048,
				hop_length = 512 # the paper says 2048, but then the output matrix is the wrong size 🤷‍♂️
			)(waveform)
			X += spectrogram
		return X	
			
def train_model(dataset):
	'''
	A CNN for classifying source seperation.
	params: dataset generated from the class above 
	'''

	print('Training neural network... 🧠')

	# configure hardware
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	plt.imshow(dataset[0][0])
	plt.show()



	# with click.progressbar(length = NUM_OF_EPOCHS) as bar:
		# for epoch in range(NUM_OF_EPOCHS):

			# bar.update(1)