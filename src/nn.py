import math, os
import torch				# pytorch
import torch.nn as nn		# neural
import torchaudio			# handling audio
import pandas as pd			# databases
import numpy as np			# maths

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
		n_mels = 128
		n_fft = 2048
		hop_length = 512 # the paper says 2048, but then the output matrix is the wrong size 🤷‍♂️
		X = torch.zeros(len(filepaths), 1, n_mels, math.ceil(4 * 44100 / hop_length))

		for i, file in enumerate(filepaths):
			waveform = torchaudio.load(os.path.join(os.getcwd(), file))[0]
			spectrogram = torchaudio.transforms.MelSpectrogram(
				sample_rate = SAMPLE_RATE,
				n_mels = n_mels,
				n_fft = n_fft,
				hop_length = hop_length
			)(waveform)
			X[i] = spectrogram

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
		y_hat, _ = self.lstm(y_hat)
		y_hat = torch.reshape(y_hat, (y_hat.shape[0], 32, 16, 43))
		y_hat = self.final_conv(y_hat)
		y_hat = torch.reshape(y_hat, (-1, 32 * 8 * 21))
		y_hat = self.fc1(y_hat)
		y_hat = self.fc2(y_hat)
		return y_hat
			
def train_model(train_dataset, test_dataset, classes):
	'''
	A CNN/LSTM for performing source seperation.
	params: 
		training dataset generate using TargetsDataset
		testing dataset generate using TargetsDataset
		classes - a list of class labels
	'''
	print('Training neural network... 🧠')

	# initialise network
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	train_loader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size = BATCH_SIZE)
	test_loader = torch.utils.data.DataLoader(dataset = test_dataset)
	model = NeuralNet(len(classes)).to(device)
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

			# update wieghts
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
					y_predicted = model(features).numpy().flatten()
					# find the most likely labels
					idx_predicted = np.argsort(y_predicted)[0 - SAMPLES_PER_TARGET :]
					# cross reference with labels
					for i in idx_predicted:
						if (labels[i].item() == 1.0):
							n_correct += 1
				accuracy = 100 * (n_correct / (10 * len(test_dataset)))
			print(f'Epoch {epoch + 1}/{NUM_OF_EPOCHS}: Loss = {loss.item():.4f}, Accuracy = {accuracy:.2f}%')

	print('Model trained! 🎛')

	return model, accuracy