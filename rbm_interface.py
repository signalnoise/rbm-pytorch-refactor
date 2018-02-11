import argparse
import torch
from tqdm import *
from torch.autograd import Variable
import rbm_pytorch
from torch.utils.data import Dataset
import numpy as np


class RBMInterface:
	"""This class wraps the module defined in rbm_pytorch to allow for fast writing of scripts to train
	new machines, whatever data set is being used.

	The general format of a program using this interface is as follows:
	-Instantiate RBMInterface
	-Call args = RBMInterface.get_args() if needed
	-Load the training data into an instance of torch.utils.data.DataLoader
	-Call initialise_output()
	-Pass the DataLoader and your choice of torch.optim optimiser and dictionary of kwargs for that
	 optimiser into RBMInterface.build_model(DataLoader, torch.optim.*, **kwargs)
	-Call RBMInterface.train()

	Attributes:
		loss_file: Reference to the file being used for data output.
		progress_bar: Reference to the tqdm instance controlling the progress bar.
		train_loader: DataLoader instance used to feed training data to the nn.
		train_op: Optimisation algorithm used in training.
		rbm: Refers to the RBM torch module.
		parser: Reference to the argument parser which interprets arguments relevant
			    to the RBM.
		args: Arguments parsed by the parser
		dtype: Stores the type of tensor that should be used by the model e.g. CPU or cuda
	"""

	def __init__(self):
		"""Instantiates the class, loads and interprets arguments.
		"""

		# These variables are used in monitoring the progress of training
		self.loss_file = None
		self.progress_bar = None

		# These variables pertain to the model itself
		self.train_loader = None
		self.train_op = None
		self.rbm = None

		# Creates an RBMParser and parses its arguments
		self.parser = RBMParser()
		self.args = self.parser.parse_args()

		# Sets the datatype being used in the torch model
		if self.args.enable_cuda:
			self.dtype = torch.cuda.FloatTensor
		else:
			self.dtype = torch.FloatTensor

		# Print the arguments and the torch library
		print(self.args)
		print("Using library:", torch.__file__)

	def train(self):
		"""Trains the model for a specified number of epochs. 
		"""

		iterations_per_epoch = len(self.train_loader)

		for epoch in self.progress_bar:

			# Define empty arrays to hold training statistics
			loss_ = np.zeros(iterations_per_epoch)
			full_reconstruction_error = np.zeros(iterations_per_epoch)
			free_energy_ = np.zeros(iterations_per_epoch)

			for i, (data, target) in enumerate(self.train_loader):

				# Cast the data in the DataLoader to the required tensor type and run train step
				data = data.type(self.dtype)

				loss_[i], full_reconstruction_error[i], free_energy_[i] = self.train_step(data)

			# Average the statistics over the epoch
			re_mean = np.mean(full_reconstruction_error)
			loss_mean = np.mean(loss_)
			free_energy_mean = np.mean(free_energy_)

			# Update the display of the progress bar and write to the output file
			self.progress_bar.set_description("Epoch {:3d} - Loss {:8.5f} - RE {:5.3g} ".format(epoch, loss_mean, re_mean))
			stats = str(epoch) + "\t" + str(loss_mean) + "\t" + str(free_energy_mean) + "\t" + str(re_mean) + "\n"
			self.loss_file.write(stats)

			# Save a state of the RBM every 10 epochs
			if epoch % 10 == 0:
				torch.save(self.rbm.state_dict(),self.args.text_output_dir + "/trained_rbm.pytorch." + str(epoch))

		# Save the final model
		torch.save(self.rbm.state_dict(), self.args.text_output_dir + "/trained_rbm.pytorch." + str(self.args.epochs))
		self.loss_file.close()

	def train_step(self, data):
		"""Defines a single training step of the model.

		Args:
			data: a torch tensor containing this iterations training data
		"""

		# Wrap the input data as an autograd variable
		data_input = Variable(data.view(-1, self.args.visible_n))

		# Run CDk steps
		new_visible, hidden, h_prob, v_prob = self.rbm(data_input)

		# Defines the loss as the difference in free energy of reconstructions to data
		data_free_energy = self.rbm.free_energy(data_input)
		loss = data_free_energy - self.rbm.free_energy(new_visible)

		# Defines the reconstruction error as the MSEloss
		reconstruction_error = self.rbm.loss(data_input, new_visible)

		# Update gradients
		self.train_op.zero_grad()
		# Manually update the gradients, do not use autograd
		self.rbm.backward(data_input, new_visible)
		self.train_op.step()

		# Returning .data[0] converts from autograd.Variable to float
		return loss.data[0], reconstruction_error.data[0], data_free_energy.data[0] 

	def build_model(self, train_loader, optimiser, **kwargs):
		"""Creates the model, interpreting some of the program arguments.
		Args:
			train_loader: An instance of torch.utils.data.DataLoader containing the training data
			optimiser: The user's choice of optimiser from torch.optim 
			**kwargs: A dictionary containing the parameters of the optimiser
		"""
		# Sets and instantiates internal variables
		self.train_loader = train_loader

		self.rbm = rbm_pytorch.RBM(k=self.args.kCD, n_vis=self.args.visible_n, n_hid=self.args.hidden_size,
		                        	enable_cuda=self.args.enable_cuda)

		self.train_op = optimiser(self.rbm.parameters(), **kwargs)

		# Load a saved state if required
		if self.args.ckpoint is not None:
			print("Loading saved network state from file", self.args.ckpoint)
			self.rbm.load_state_dict(torch.load(self.args.ckpoint))

	def initialise_output(self, **kwargs):
		"""Creates internal variables, opens the output file so it can be written to
		Args:
			**kwargs: A dictionary of keys and values specifying the hyperparameters
					  of the training.
		"""
		self.progress_bar = tqdm(range(self.args.start_epoch, self.args.epochs))

		filename = self.args.text_output_dir + "/Loss_timeline"

		for key, value in kwargs.items():
			filename = filename + "_" + key + "_" + str(value)

		filename = filename + ".data"

		header = "#Epoch \t  Loss mean \t free energy mean \t reconstruction error mean \n"

		self.loss_file = open(filename, "w", buffering=1)
		self.loss_file.write(header)

	def get_args(self):
		"""Returns the arguments parsed by the RBMParser.
		"""

		return self.args

class RBMParser(argparse.ArgumentParser):
	"""Extends ArgumentParser with predefined arguments specific to the RBM.
	"""

	def __init__(self):
		"""Instantiates the object and defines arguments for the RBM.
		"""
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
		self.add_argument('--training_data', dest='training_data', help='path to training input data',
		                         type=str)
		self.add_argument('--txtout', dest='text_output_dir', default='./',
		                         help='Directory in which to save text output data',
		                         type=str)
		self.add_argument('--enable_cuda', dest='enable_cuda', default=False, help='Specifies if cuda will be used',
		                  type=bool)

class CSV_Ising_dataset(Dataset):
    """ Extends Dataset to interpret the output format of magneto.
    """

    def __init__(self, csv_file, size=32, transform=None):
        self.csv_file = csv_file
        self.size = size
        csvdata = np.loadtxt(csv_file, delimiter=",", skiprows=1, dtype="float64")
        self.imgs = torch.from_numpy(csvdata.reshape(-1, size))
        self.datasize, sizesq = self.imgs.shape
        self.transform = transform
        print("Loaded training set of %d states" % self.datasize)

    def __getitem__(self, index):
        return self.imgs[index], index

    def __len__(self):
        return len(self.imgs)