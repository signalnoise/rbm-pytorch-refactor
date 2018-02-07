**rbm-pytorch-refactor**
-------------------------
This is a refactor of the project rbm_ising. It extends the interface of the program, allowing it to be used with ANY kind of data as long as you can get it in a DataLoader.

The general format of a program using this interface is as follows:
* Instantiate RBMInterface
* Call args = RBMInterface.get_args() if needed
* Load the training data into an instance of torch.utils.data.DataLoader
* Call initialise_output()
* Pass the DataLoader and your choice of torch.optim optimiser and dictionary of kwargs for that optimiser into RBMInterface.build_model(DataLoader, torch.optim.*, **kwargs)
* Call RBMInterface.train()
