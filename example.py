import rbm_interface
from torch.utils.data import DataLoader
import torch.optim

rbm = rbm_interface.RBMInterface(use_validation=True)
args = rbm.get_args()

training_set, validation_set, comparison_set = rbm_interface.ising_loader(args.training_data, size=args.visible_n, validation_size=10000).get_datasets()


rbm.initialise_output()

rbm.build_model(args.batches, training_set, torch.optim.Adadelta, validation_set,
				comparison_set)#, lr=0.01)

rbm.train()
