import os
import torch				# pytorch
import torch.nn as nn		# neural
import torchaudio			# handling audio
import pandas as pd			# databases
import numpy as np			# maths
import librosa				# audio tools

from settings import BATCH_SIZE, LEARNING_RATE, NUM_OF_EPOCHS, SAMPLE_RATE, SAMPLES_PER_TARGET

class TargetsDataset(torch.utils.data.Dataset):
	'''
	The dataset
		self.X is a tensor of mel spectrograms
		self.Y is a tensor of logits for each label
	'''
	def __init__(self, targets, num_of_classes):
		# transform json into np arrays and preprocess
		self.X = self.preprocess_x(pd.DataFrame(targets)['filepath'].values)
		self.Y = self.preprocess_y(pd.DataFrame(targets)['labels'].values, num_of_classes)
		self.n_samples = self.Y.shape[0]

	# helper methods
	def __getitem__(self, index):
		return self.X[index], self.Y[index]
	def __len__(self):
		return self.n_samples

	# generate mel spectrograms from rendered targets
	def preprocess_x(self, filepaths):
		tmp = []
		for i, file in enumerate(filepaths):
			tmp += [import_to_mel(os.path.join(os.getcwd(), file), SAMPLE_RATE)]
		X = torch.stack(tmp)
		X.requires_grad = True
		return X

	# converts np array of lists to a tensor
	def preprocess_y(self, array_of_lists, num_of_classes):
		Y = torch.zeros([len(array_of_lists), num_of_classes])
		for i, list in enumerate(array_of_lists):
			for index in list:
				Y[i][index] = 1
		return Y

class ConvLayer(nn.Module):
	'''
	Convolutional layer -> batch norm -> relu -> max pool
	'''
	def __init__(self, input_size, output_size):
		super(ConvLayer, self).__init__()
		self.conv = nn.Conv2d(input_size, output_size, 3, stride = 1, padding = 1)
		self.bn = nn.BatchNorm2d(output_size)
		self.pool = nn.MaxPool2d(2, stride = 2)

	def forward(self, x):
		return self.pool(nn.functional.relu(self.bn(self.conv(x))))

class NeuralNet(nn.Module):
	'''
	Three conv layers -> lstm -> conv layer -> 2 fc layers
	'''
	def __init__(self, num_of_classes):
		super(NeuralNet, self).__init__()
		self.convLayers = nn.ModuleList([ConvLayer(1, 8), ConvLayer(8, 16), ConvLayer(16, 32)])
		self.lstm = nn.LSTM(16*43, 16*43, batch_first = True)
		self.final_conv = ConvLayer(32, 32)
		self.fc1 = nn.Linear(32 * 8 * 21, num_of_classes)
		self.fc2 = nn.Linear(num_of_classes, num_of_classes)

	def forward(self, x):
		y_hat = x
		for i in range(3):
			y_hat = self.convLayers[i](y_hat)
		y_hat = torch.flatten(y_hat, start_dim=2)
		y_hat = self.lstm(y_hat)[0]
		y_hat = torch.reshape(y_hat, (y_hat.shape[0], 32, 16, 43))
		y_hat = self.final_conv(y_hat)
		y_hat = torch.flatten(y_hat, start_dim=1)
		y_hat = self.fc1(y_hat)
		y_hat = self.fc2(y_hat)
		return y_hat
			
def import_to_mel(filepath, sample_rate):
	'''
	Import target audio file and pre process.
	input:
		filepath to target audio
		sample_rate of the current model
	output:
		a mel spectrogram of the loaded audiofile, normalised and limited to 4 seconds. 
	'''
	# mel settings
	n_mels = 128
	n_fft = 2048
	hop_length = 512 # the paper says 2048, but then the output matrix is the wrong size 🤷‍♂️

	# import, convert to mono, normalise
	waveform, sr = torchaudio.load(filepath)
	waveform = waveform.numpy()
	if (sr != sample_rate):
		waveform = librosa.resample(waveform, sr, sample_rate)
	if (waveform.shape[0] > 1):
		waveform = librosa.to_mono(waveform).reshape(1, len(waveform[0]))
	waveform[0] = waveform[0] * (1.0 / np.max(waveform[0])) # normalise
	waveform = torch.from_numpy(waveform)

	# copy to a tensor of specific size
	waveform_4s = torch.zeros(1, sample_rate * 4)
	iter_len = min(sample_rate * 4, waveform.shape[1])
	for i in range(iter_len):
		waveform_4s[0][i] = waveform[0][i]

	# generate mel
	spectrogram = torchaudio.transforms.MelSpectrogram(
		sample_rate = sample_rate,
		n_mels = n_mels,
		n_fft = n_fft,
		hop_length = hop_length
	)(waveform_4s)
	return spectrogram

def train_model(train_dataset, test_dataset, num_of_classes):
	'''
	A CNN/LSTM for performing source seperation.
	params: 
		training dataset generate using TargetsDataset
		testing dataset generate using TargetsDataset
		number of output classes
	'''
	print('Training neural network... 🧠')

	# initialise network
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	train_loader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size = BATCH_SIZE)
	test_loader = torch.utils.data.DataLoader(dataset = test_dataset)
	model = NeuralNet(num_of_classes).to(device)
	criterion = nn.BCEWithLogitsLoss()
	optimiser = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

	# training loop
	for epoch in range(NUM_OF_EPOCHS):
		for i, (features, labels) in enumerate(train_loader):
			# load data to the gpu / cpu
			features = features.to(device)
			labels = labels.to(device)
			
			# calculate predictions and loss
			outputs = model(features)
			loss = criterion(outputs, labels)

			# update weights
			optimiser.zero_grad()
			loss.backward()
			optimiser.step()

		if (epoch == 0 or epoch % 5 == 4):
			# test model
			with torch.no_grad():
				n_correct = 0
				for (features, labels) in test_loader:
					features = features.to(device)
					labels = torch.flatten(labels)
					# return output as a numpy array
					y_predicted = model(features).cpu()
					y_predicted = y_predicted.numpy().flatten()
					# find the most likely labels
					idx_predicted = np.argsort(y_predicted)[0 - SAMPLES_PER_TARGET :]
					# cross reference with labels
					for i in idx_predicted:
						if (labels[i].item() == 1.0):
							n_correct += 1
				accuracy = 100 * (n_correct / (SAMPLES_PER_TARGET * len(test_dataset)))
			print(f'Epoch {epoch + 1}/{NUM_OF_EPOCHS}: Loss = {loss.item():.4f}, Accuracy = {accuracy:.2f}%')

	print('Model trained! 🎛')

	return model, accuracy