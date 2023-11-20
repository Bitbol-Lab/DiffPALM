

import numpy as np

from diffpalm.core import DiffPALM
from diffpalm.msa_parsing import read_msa
from diffpalm.datasets import generate_dataset, dataset_tokenizer

from pathlib import Path
from datetime import datetime
import pickle


def save_parameters(parameters_all, filepath):
	"""Saves the parameters dictionary"""
	for name, parameters in parameters_all.items():
		with open(filepath / f"{name}.pkl", "wb") as f:
			pickle.dump(parameters, f)
		with open(filepath / f"{name}.csv", "w") as f:
			for key in parameters.keys():
				f.write("%s, %s\n" % (key, parameters[key]))


# where to define this?
get_species_name = (lambda strn: strn.split("|")[1])

def read_data (data_flag = "HK-RR"):

	data_in_dir = "/app/data" # TO DO - docker interface change this
	# TO DO - form the input data path + pass it on

	if data_flag == "HK-RR":
		msa_data = [read_msa(data_in_dir+"/HK-RR/HK_in_Concat_nnn.fasta", -1),
			read_msa(data_in_dir+"/HK-RR/RR_in_Concat_nnn.fasta", -1)]

	return msa_data



if __name__ == "__main__":

	EPOCHS = 100
	TORCH_SEED = 100

	DOCKER_SHARE_BASE_DIR = "/app/data/"
	run_date1 = datetime.now().strftime("%Y_%b_%d")
	RESULTS_DIR = Path (DOCKER_SHARE_BASE_DIR+run_date1)
	RESULTS_DIR.mkdir(exist_ok=True)
	
	# set it based on gpu availability?
	DEVICE  = 'cuda' # is this needed?

	# define various parameters
	# TO DO : make them into ARGS for invocation via Docker
	# or read from a config file. TBD

	parameters_dataset = {
		"N": 50,  # Average number of sequences in the input
		"pos": 0,  # Size of the context pairs to use as positive example 
		"max_size": 100,  # Max size of species MSAs (if same as N there is no limit on size)
		"NUMPY_SEED": 10,
		"NUMPY_SEED_OTHER": 11,
}
	parameters_init = {
	"device": DEVICE,
	"p_mask": 0.7,
	"random_seed": TORCH_SEED
}

	parameters_train = {
		"std_init": 0.,
		"scheduler_name": "ReduceLROnPlateau",
		"scheduler_kwargs": {"mode": "min", "factor": 0.8, "patience": 20},
		"optimizer_name": "Adadelta",
		"optimizer_kwargs": {"lr": 9, "weight_decay": 1e-1},
		"tau": 1.,
		"n_sink_iter": 10,
		"batch_size": 1,
		"epochs": EPOCHS,
		"noise": True,
		"noise_factor": 0.1,  # If noise_std is False, this is just the std of the noise
		"noise_scheduler": True,
		"noise_std": True,
		"use_rand_perm": True,
	}

	parameters_target_loss = {
	"batch_size": 200
}

	parameters_all = {
	"init": parameters_init,
	"target_loss": parameters_target_loss,
	"train": parameters_train,
	"dataset": parameters_dataset
}

	# save parameters for keeping track of texperiments
	date = datetime.now().strftime("%Y_%m_%d-%H:%M:%S")
	output_dir = RESULTS_DIR / date   # TO DO : NOT YET INITIALIZED
	output_dir.mkdir() 
	save_parameters(parameters_all, output_dir)


	msa_data = read_data()

	# preprocess and clean up data
	get_species_name = (lambda strn: strn.split("|")[1])

	dataset, species_sizes = generate_dataset(
		parameters_dataset, msa_data, get_species_name=get_species_name
		)
	tokenized_dataset = dataset_tokenizer(dataset, device=DEVICE)
	left_msa, right_msa = tokenized_dataset["msa"]["left"], tokenized_dataset["msa"]["right"]
	positive_examples = tokenized_dataset["positive_examples"]
	# initialize the model
	dpalm = DiffPALM(species_sizes, **parameters_init)

	save_all_figs=True # do we want to save it -give it as an option
	tar_loss = dpalm.target_loss(
			left_msa,
			right_msa,
			positive_examples=positive_examples,
			**parameters_target_loss
		)

	(losses, 
		list_scheduler, 
		shuffled_indexes, 
		mat_perm, 
		mat_gs, 
		list_log_alpha) = dpalm.train(
							left_msa,
							right_msa,
							positive_examples=positive_examples,
							tar_loss=np.mean(tar_loss),
							output_dir=output_dir,
							save_all_figs=True,
							**parameters_train,
							)
	results = {
			"trainng_results": (losses, list_scheduler, shuffled_indexes, [mat_perm, mat_gs], list_log_alpha),
			"target_loss": tar_loss,
			"species_sizes": species_sizes
			}
	 

