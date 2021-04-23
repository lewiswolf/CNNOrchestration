import math, os
import torch				# pytorch
import torch.nn as nn		# neural
import torchaudio			# handling audio
import pandas as pd			# databases

from settings import BATCH_SIZE, LEARNING_RATE, NUM_OF_EPOCHS, SAMPLE_RATE

class TargetsDataset(torch.utils.data.Dataset):
	'''
	The dataset
		self.X is a tensor of mel spectrograms
		self.Y is a tensor of class labels.
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
		self.bn = nn.BatchNorm2d(input_size)
		self.pool = nn.MaxPool2d(2, stride = 2)

	def forward(self, x):
		return self.pool(nn.functional.relu(self.bn(self.conv(x))))

class LSTM(nn.Module):
	'''
	LSTM that produces an output from each layer
	definition paramets:
		input size,
		hidden size,
		number of layers
	input tensor:
		(sequence, batch number, features)
	output tensor:
		(batch number, lstm layer, sequence, features)
	'''
	def __init__(self, input_size, hidden_size, num_layers):
		super(LSTM, self).__init__()
		self.num_layers = num_layers
		self.LSTMLayers = nn.ModuleList([nn.LSTM(input_size, hidden_size, 1) for i in range(num_layers)])
		pass

	def forward(self, X):
		h = torch.randn(1, X.shape[1], X.shape[2])
		c = torch.randn(1, X.shape[1], X.shape[2])
		y_hat = []
		for i in range(self.num_layers):
			out, (h, c) = self.LSTMLayers[i](X, (h, c))
			y_hat += torch.reshape(out, (-1, X.shape[0], X.shape[1], X.shape[2]))
		y_hat = torch.stack(y_hat)
		y_hat = torch.transpose(y_hat, 1, 2)
		y_hat = torch.transpose(y_hat, 0, 1)
		return y_hat, (h, c)

class NeuralNet(nn.Module):
	'''
	Three conv layers -> lstm -> conv layer -> 2 fc layers
	'''
	def __init__(self, num_of_classes):
		super(NeuralNet, self).__init__()
		self.convLayers = nn.ModuleList([ConvLayer(1, 1) for i in range(3)])
		self.lstm = LSTM(43, 43, 32)
		self.final_conv = ConvLayer(32, 32)
		self.fc1 = nn.Linear(32 * 8 * 21, num_of_classes)
		self.fc2 = nn.Linear(num_of_classes, num_of_classes)

	def forward(self, x):
		y_hat = x
		for i in range(3):
			y_hat = self.convLayers[i](y_hat)
		y_hat = torch.transpose(torch.reshape(y_hat, (-1, 16, 43)), 0, 1)
		y_hat, _ = self.lstm(y_hat)
		y_hat = self.final_conv(y_hat)
		y_hat = torch.reshape(y_hat, (-1, 32 * 8 * 21))
		y_hat = self.fc1(y_hat)
		y_hat = self.fc2(y_hat)
		return y_hat
			
def train_model(train_dataset, test_dataset, classes):
	'''
	A CNN/LSTM for performing source seperation.
	params: 
		dataset - generated from the class above 
		classes - a list of class variables
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

		if (epoch % 5 == 0):
			print(f'Epoch {epoch}/{NUM_OF_EPOCHS}: Loss = {loss.item():.4f}')

	print('Model trained! 🎛')

	# test model
	with torch.no_grad():
		accuracy = 0
		for (features, labels) in test_loader:
			features = features.to(device)
			labels = labels.to(device)
			outputs = model(features)
	
	return model, accuracy