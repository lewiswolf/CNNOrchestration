import os
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
		print('Preprocessing dataset... 📝')
		# transform json into np arrays and preprocess
		self.X = self.preprocess_x(pd.DataFrame(targets)['filepath'].values)
		self.Y = self.preprocess_y(pd.DataFrame(targets)['labels'].values, num_of_classes)
		self.n_samples = self.Y.shape[0]

	# helper methods
	def __getitem__(self, index):
		return self.X[index], self.Y[index]
	def __len__(self):
		return self.n_samples

	# generate spectrograms from rendered targets
	# and return a tensor of size (NUM_OF_TARGETS, 1, n_mels, (4 * 44100 / 2048)))
	def preprocess_x(self, filepaths):
		tmp = []
		for x in filepaths:
			waveform = torchaudio.load(os.path.join(os.getcwd(), x))[0]
			spectrogram = torchaudio.transforms.MelSpectrogram(
				sample_rate = SAMPLE_RATE,
				n_mels = 128,
				n_fft = 2048,
				hop_length = 512 # the paper says 2048, but then the output matrix is the wrong size 🤷‍♂️
			)(waveform)
			tmp += spectrogram
		X = torch.reshape(torch.stack(tmp), (-1, 1, 128, 345))
		X.requires_grad = True
		return X

	# converts np array to a list then to a tensor of size (NUM_OF_TARGETS, num_of_classes)
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
	def __init__(self):
		super(ConvLayer, self).__init__()
		self.conv = nn.Conv2d(1, 1, 3, stride = 1, padding = 1)
		self.bn = nn.BatchNorm2d(1)
		self.pool = nn.MaxPool2d(2, stride = 2)

	def forward(self, x):
		return self.pool(nn.functional.relu(self.bn(self.conv(x))))

class NeuralNet(nn.Module):
	'''
	Three conv layers -> lstm -> conv layer -> 2 fc layers -> sigmoid
	'''
	def __init__(self, num_of_classes):
		super(NeuralNet, self).__init__()
		self.convLayers = nn.ModuleList([ConvLayer() for i in range(4)])
		self.fc1 = nn.Linear(8 * 21, num_of_classes) # ammend inputs when lstm is involved
		self.fc2 = nn.Linear(num_of_classes, num_of_classes)
		self.a = nn.Sigmoid()

	def forward(self, x):
		y_hat = x
		for i in range(3):
			y_hat = self.convLayers[i](y_hat)

		y_hat = self.convLayers[3](y_hat)
		y_hat = torch.reshape(y_hat, (-1, 8 * 21)) # ammend when lstm is involved
		y_hat = self.fc1(y_hat)
		y_hat = self.fc2(y_hat)
		y_hat = self.a(y_hat)
		return y_hat

			
def train_model(dataset, classes):
	'''
	A CNN/LSTM for classifying source seperation.
	params: 
		dataset - generated from the class above 
		classes - a list of class variables
	'''

	print('Training neural network... 🧠')

	# configure network settings
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	dataloader = torch.utils.data.DataLoader(dataset = dataset, batch_size = BATCH_SIZE)
	model = NeuralNet(len(classes)).to(device)
	criterion = nn.BCELoss()
	optimiser = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

	for epoch in range(NUM_OF_EPOCHS):
		for i, (features, labels) in enumerate(dataloader):
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