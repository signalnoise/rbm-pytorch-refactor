import sys
sys.path.append('../Refactor/')
sys.path.append('../rbm-pytorch-refactor/')
import rbm_pytorch

def NLL_estimate(rbm, train_loader):	
	sum_free_energy = 0
	for i, (data, target) in enumerate(self.train_loader):
		sum_free_energy = sum_free_energy + rbm.free_energy(v, size_average=False).sum().data[0]
	logZ, highZ, lowZ = rbm.annealed_importance_sampling()
	avg_f_e = sum_free_energy/n_samples
	nll = logZ + avg_f_e
	upper_bound = (highZ) + avg_f_e
	lower_bound = (lowZ) + avg_f_e
	return (nll, upper_bound, lower_bound)

def LL_estimate(rbm, train_loader):
		sum_free_energy = 0
	for i, (data, target) in enumerate(self.train_loader):
		sum_free_energy = sum_free_energy + rbm.free_energy(v, size_average=False).sum().data[0]
	logZ, highZ, lowZ = rbm.annealed_importance_sampling()
	avg_f_e = sum_free_energy/n_samples
	ll = -logZ - avg_f_e
	upper_bound = -(highZ) - avg_f_e
	lower_bound = -(lowZ) - avg_f_e
	return (ll, upper_bound, lower_bound)