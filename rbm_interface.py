import argparse
import torch
from tqdm import *
from torch.autograd import Variable
import rbm_pytorch
from torch.utils.data import Dataset
import numpy as np


class RBMInterface:

	def __init__(self):

		self.loss_file = None
		self.progress_bar = None

		self.train_loader = None
		self.train_op = None
		self.rbm = None

		self.parser = rbm_parser()
		self.args = self.parser.parse_args()

		print(self.args)
		print("Using library:", torch.__file__)

	def train(self):

		iterations_per_epoch = len(self.train_loader)

		for epoch in self.progress_bar:

			loss_ = np.zeros(iterations_per_epoch)
			full_reconstruction_error = np.zeros(iterations_per_epoch)
			free_energy_ = np.zeros(iterations_per_epoch)

			for i, (data, target) in enumerate(self.train_loader):

				loss_[i], full_reconstruction_error[i], free_energy_[i] = self.train_step(data)

			re_mean = np.mean(full_reconstruction_error)
			loss_mean = np.mean(loss_)
			free_energy_mean = np.mean(free_energy_)

		self.progress_bar.set_description("Epoch %3d - Loss %8.5f - RE %5.3g " % (epoch, loss_mean, re_mean))

		stats = str(epoch) + "\t" + str(loss_mean) + "\t" + str(free_energy_mean) + "\t" + str(re_mean) + "\n"
		self.loss_file.write(stats)

		if epoch % 10 == 0:
			torch.save(self.rbm.state_dict(), "trained_rbm.pytorch." + str(epoch))

		# Save the final model
		torch.save(self.rbm.state_dict(), "trained_rbm.pytorch." + str(self.args.epochs))
		self.loss_file.close()

	def train_step(self, data):

		data_input = Variable(data.view(-1, self.args.visible_n))

		new_visible, hidden, h_prob, v_prob = self.rbm(data_input)

		data_free_energy = self.rbm.free_energy(data_input)
		loss = data_free_energy - self.rbm.free_energy(new_visible)

		reconstruction_error = self.rbm.loss(data_input, new_visible)

		# Update gradients
		self.train_op.zero_grad()
		# manually update the gradients, do not use autograd
		self.rbm.backward(data_input, new_visible)
		self.train_op.step()

		# Returning .data[0] converts from autograd.Variable to float
		return loss.data[0], reconstruction_error.data[0], data_free_energy.data[0]

	def build_model(self, train_loader, optimiser, **kwargs):
		self.train_loader = train_loader

		if self.args.enable_cuda:
			self.rbm = rbm_pytorch.RBM(k=self.args.kCD, n_vis=self.args.visible_n, n_hid=self.args.hidden_size,
			                           enable_cuda=self.args.enable_cuda).cuda()
		else:
			self.rbm = rbm_pytorch.RBM(k=self.args.kCD, n_vis=self.args.visible_n, n_hid=self.args.hidden_size,
			                           enable_cuda=self.args.enable_cuda)

		if self.args.ckpoint is not None:
			print("Loading saved network state from file", self.args.ckpoint)
			self.rbm.load_state_dict(torch.load(self.args.ckpoint))

		self.train_op = optimiser(self.rbm.parameters(), **kwargs)

	def initialise_output(self, **kwargs):
		self.progress_bar = tqdm(range(self.args.start_epoch, self.args.epochs))

		filename = self.args.text_output_dir + "Loss_timeline_"

		for key, value in kwargs.items():
			filename = filename + key + "_" + str(value) + "_"

		header = "#Epoch \t  Loss mean \t free energy mean \t reconstruction error mean \n"

		self.loss_file = open(filename, "w", buffering=1)
		self.loss_file.write(header)

	def get_args(self):

		return self.args

class rbm_parser(argparse.ArgumentParser):

	def __init__(self):
		super().__init__(description='Process arguments for a Restricted Boltzmann Machine')

		self.add_argument('--ckpoint', dest='ckpoint', help='Path to RBM saved state',
		                         type=str)
		self.add_argument('--start', dest='start_epoch', default=0, help='Choose the first epoch number',
		                         type=int)
		self.add_argument('--epochs', dest='epochs', default=10, help='number of epochs',
		                         type=int)
		self.add_argument('--batch', dest='batches', default=100, help='batch size',
		                         type=int)
		self.add_argument('--hidden', dest='hidden_size', default=64, help='Number of hidden nodes',
		                         type=int)
		self.add_argument('--visible', dest='visible_n', default=64, help='Number of visible nodes',
		                  type=int)

		self.add_argument('--k', dest='kCD', default=2, help='number of Contrastive Divergence steps',
		                         type=int)
		self.add_argument('--training_data', dest='training_data', default='../../state0t2.1.txt',
		                         help='path to training input data',
		                         type=str)
		self.add_argument('--txtout', dest='text_output_dir', default='./',
		                         help='Directory in which to save text output data',
		                         type=str)
		self.add_argument('--enable_cuda', dest='enable_cuda', default=False, help='Specifies if cuda will be used',
		                  type=bool)

class CSV_Ising_dataset(Dataset):
    """ Defines a CSV reader """
    def __init__(self, csv_file, size=32, transform=None):
        self.csv_file = csv_file
        self.size = size
        csvdata = np.loadtxt(csv_file, delimiter=",", skiprows=1, dtype="float32")
        self.imgs = torch.from_numpy(csvdata.reshape(-1, size))
        self.datasize, sizesq = self.imgs.shape
        self.transform = transform
        print("Loaded training set of %d states" % self.datasize)

    def __getitem__(self, index):
        return self.imgs[index], index

    def __len__(self):
        return len(self.imgs)