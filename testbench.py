#%%

import pycatch22
import os
import csv
import numpy as np

#%%

tsData = [1,2,4,3,5,1,2,5]

features = pycatch22.catch22_all(tsData)["values"]

print(features)

#%%

import pycatch22
import os
import csv
import numpy as np

def get_label(filename):

    for i, char in enumerate(filename):
        if char == "-":
            return filename[(i+1):(i+4)]
        
data_path = r"C:\Users\ronny\Documents\GitHub\MetaPain\Data\biosignals_filtered"
folders = os.listdir(data_path)

ID = os.path.join(data_path, "082814_w_46")
files = os.listdir(ID)
data = []
for file in files:
    with open(os.path.join(ID, file),'r') as f:
        pickle_in = list(csv.reader(f, delimiter='\t'))
        print(file+" read done")
        seq = np.array(pickle_in)[1:,3:6].astype(float)
        features1 =  pycatch22.catch22_all(list(seq[:,0]))["values"]
        features2 =  pycatch22.catch22_all(list(seq[:,1]))["values"]
        features3 =  pycatch22.catch22_all(list(seq[:,2]))["values"]
        print(file+" process done")
        features = [*features1, *features2, *features3]
        features.append(get_label(file))
        data.append(features)

#%%
import pycatch22
import os
import csv
import numpy as np

def get_label(filename):

    for i, char in enumerate(filename):
        if char == "-":
            return filename[(i+1):(i+4)]

data_path = r"C:\Users\ronny\Documents\GitHub\MetaPain\Data\biosignals_filtered"
folders = os.listdir(data_path)
ID = os.path.join(data_path, "082814_w_46")
files = os.listdir(ID)
file = "082814_w_46-BL1-099_bio.csv"
with open(os.path.join(ID, file),'r') as f:
    pickle_in = list(csv.reader(f, delimiter='\t'))
    seq = np.array(pickle_in)[1:,3:6].astype(float)
    features1 =  pycatch22.catch22_all(list(seq[:,0]))["values"]
    features2 =  pycatch22.catch22_all(list(seq[:,1]))["values"]
    features3 =  pycatch22.catch22_all(list(seq[:,2]))["values"]
    features = [*features1, *features2, *features3]
    features.append(get_label(file))
    

#%% Preprocessing Script
def get_label(filename):

    for i, char in enumerate(filename):
        if char == "-":
            return filename[(i+1):(i+4)]

def processing_file(data_folder, dump_location = None, mode = "fusion"):

    folders = os.listdir(data_folder)
    for folder in folders:

        ID = os.path.join(data_folder, folder)
        files = os.listdir(ID)
        data = []

        if mode == "fusion":
            for file in files:
                with open(os.path.join(ID, file),'r') as f:
                    pickle_in = list(csv.reader(f, delimiter='\t'))
                    seq = np.array(pickle_in)[1:,1:].astype(float)
                    eda =  pycatch22.catch22_all(list(seq[:,0]))["values"]
                    ecg =  pycatch22.catch22_all(list(seq[:,1]))["values"]
                    emg1 =  pycatch22.catch22_all(list(seq[:,2]))["values"]
                    emg2 =  pycatch22.catch22_all(list(seq[:,3]))["values"]
                    emg3 =  pycatch22.catch22_all(list(seq[:,4]))["values"]
                    features = [*eda, *ecg, *emg1, *emg2, *emg3]
                    features.append(get_label(file))
                    data.append(features)

        elif mode == "eda":
            for file in files:
                with open(os.path.join(ID, file),'r') as f:
                    pickle_in = list(csv.reader(f, delimiter='\t'))
                    seq = np.array(pickle_in)[1:,1].astype(float)
                    features =  pycatch22.catch22_all(list(seq))["values"]
                    features.append(get_label(file))
                    data.append(features)

        elif mode == "ecg":
            for file in files:
                with open(os.path.join(ID, file),'r') as f:
                    pickle_in = list(csv.reader(f, delimiter='\t'))
                    seq = np.array(pickle_in)[1:,2].astype(float)
                    features =  pycatch22.catch22_all(list(seq))["values"]
                    features.append(get_label(file))
                    data.append(features)
                    
        elif mode == "emg":
            for file in files:
                with open(os.path.join(ID, file),'r') as f:
                    pickle_in = list(csv.reader(f, delimiter='\t'))
                    seq = np.array(pickle_in)[1:,3:6].astype(float)
                    features1 =  pycatch22.catch22_all(list(seq[:,0]))["values"]
                    features2 =  pycatch22.catch22_all(list(seq[:,1]))["values"]
                    features3 =  pycatch22.catch22_all(list(seq[:,2]))["values"]
                    features = [*features1, *features2, *features3]
                    features.append(get_label(file))
                    data.append(features)
        else:
            raise ValueError("You did not provide a supported feature class")

        if dump_location is None:
            raise ValueError("You need to provide a file dumping location")
        loc = dump_location+"/"+folder+"_"+mode+".csv"
        np.savetxt(loc, np.array(data), delimiter=",", fmt='%s')
        
#%%

data_path = r"C:\Users\ronny\Documents\GitHub\MetaPain\Data\biosignals_filtered"
dump_path = "C:/Users/ronny/Documents/GitHub/MetaPain/Data/biosignals_filtered_Processed/ecg"
processing_file(data_folder = data_path, dump_location = dump_path, mode = "ecg")

#%%

data_path = r"C:\Users\ronny\Documents\GitHub\MetaPain\Data\biosignals_filtered"
dump_path = "C:/Users/ronny/Documents/GitHub/MetaPain/Data/biosignals_filtered_Processed/emg"
processing_file(data_folder = data_path, dump_location = dump_path, mode = "emg")

#%%

data_path = r"C:\Users\ronny\Documents\GitHub\MetaPain\Data\biosignals_filtered"
dump_path = "C:/Users/ronny/Documents/GitHub/MetaPain/Data/biosignals_filtered_Processed/eda"
processing_file(data_folder = data_path, dump_location = dump_path, mode = "eda")

#%%

data_path = r"C:\Users\ronny\Documents\GitHub\MetaPain\Data\biosignals_filtered"
dump_path = "C:/Users/ronny/Documents/GitHub/MetaPain/Data/biosignals_filtered_Processed/fusion"
processing_file(data_folder = data_path, dump_location = dump_path, mode = "fusion")

#%% Test Bench for the variational random features model

import os
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from stable_sgd.models import MLP, ResNet18, ResNet18MiniImagenet
from stable_sgd.data_utils import get_permuted_mnist_tasks, get_rotated_mnist_tasks, get_split_cifar100_tasks, get_mini_imagenet_tasks, get_train_loader, get_test_loader, coreset_selection, iterate_data
from stable_sgd.utils import parse_arguments, DEVICE, init_experiment, end_experiment, log_metrics, log_hessian, save_checkpoint, sample, rand_features, dotp_kernel, kl_div, cosine_dist, AverageMeter
import math
import pdb
import time as benchmarker

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train_single_epoch(net, optimizer, loader, coreset_loader, criterion, bias, lmd, gma, bta, args, task_id=None):
	"""
	Train the model for a single epoch
	
	:param net:
	:param optimizer:
	:param loader:
	:param criterion:
	:param task_id:
	:return:
	"""
	net = net.to(DEVICE)
	net.train()

	coreset_input, coreset_target = iterate_data(coreset_loader)
	coreset_input  = coreset_input.to(DEVICE)
	coreset_target = torch.nn.functional.one_hot(coreset_target, num_classes=10 if "mnist" in args.dataset else 100)
	coreset_target = coreset_target.to(DEVICE).float()
	
	task_acc = AverageMeter()
	task_loss = AverageMeter()

	time_complexity_history = AverageMeter()

	# import pdb; pdb.set_trace()
	for _, (data, target) in enumerate(loader):
		start_time = benchmarker.time()
		data = data.to(DEVICE)
		target = target.to(DEVICE)
		batch_size = data.shape[0]
		optimizer.zero_grad()
		
		if task_id:
			features_train,   p_mu, p_log_var = net(data, task_id, post = False, predefined=True)
			features_coreset, r_mu, r_log_var  = net(coreset_input, task_id, post = True)
		else:
			features_train,   p_mu, p_log_var = net(data,          post = False, predefined=True)
			features_coreset, r_mu, r_log_var  = net(coreset_input, post = True)

		# compute_bases
		rs = sample(r_mu, r_log_var, args.d_rn_f, DEVICE).squeeze()


		# Here we map the coreset data into random fourier space. 
		x_supp_phi_t = rand_features(rs, torch.transpose(features_coreset, 1, 0), bias)  # (d_rn_f , number of coreset samples)
		# in the random fourier space we compute the the kernel matrix, the dot product between mapped samples
		support_kernel = dotp_kernel(torch.transpose(x_supp_phi_t, 1, 0), x_supp_phi_t)  # (number of coreset samples , number of coreset samples)

		# closed-form solution with trainable lmd
		# alpha = torch.matmul(torch.inverse(support_kernel + (lmd_abs + 0.01) * torch.eye(support_kernel[0])), target)
		# (number of coreset samples, number of classes)
		alpha = torch.matmul(torch.inverse(support_kernel + (torch.abs(lmd[task_id-1]) + 0.01) * torch.eye(support_kernel.shape[0]).to(DEVICE)), coreset_target) 
		
		# Here we map the train data into random fourier space. 
		x_que_phi_t = rand_features(rs, torch.transpose(features_train, 1, 0), bias)

		# Compute cross kernel
		cross_kernel = dotp_kernel(torch.transpose(x_supp_phi_t, 1, 0), x_que_phi_t) # (number of coreset samples, number of train samples)

		pred = gma[task_id-1] * torch.matmul(cross_kernel.T, alpha) + bta[task_id-1]
		
		end_time = benchmarker.time()

		time_complexity_history.update((end_time - start_time)/batch_size)

		# import pdb; pdb.set_trace()
		if "cifar" in args.dataset or "mini" in args.dataset:
			t = task_id
			offset1 = int((t-1) * 5)
			offset2 = int(t * 5)
			
			# offset1 = class_ids[task_id-1][0]
			# offset2 = class_ids[task_id-1][1]
			
			if offset1 > 0:
				pred[:, :offset1].data.fill_(-10e10)
			if offset2 < 100:
				pred[:, offset2:100].data.fill_(-10e10)



		kl_loss = kl_div(r_mu, r_log_var, p_mu, p_log_var)
		
		loss = criterion(pred, target) + args.tau * kl_loss
		task_loss.update(loss.item())

		acc = (pred.argmax(dim=-1) == target).float().mean()
		task_acc.update(acc.item())


		loss.backward()
		optimizer.step()
	
	print(f"Task {task_id} time complexity is {time_complexity_history.avg}")
	global time_complexity
	time_complexity.update(time_complexity_history.avg)

	print(task_acc.avg, task_loss.avg)
	return net


def eval_single_epoch(net, loader, coreset_loader, criterion, bias, lmd, gma, bta, args, task_id=None):
	"""
	Evaluate the model for single epoch
	
	:param net:
	:param loader:
	:param criterion:
	:param task_id:
	:return:
	"""

	net = net.to(DEVICE)
	net.eval()

	coreset_input, coreset_target = iterate_data(coreset_loader)
	coreset_input  = coreset_input.to(DEVICE)
	coreset_target = torch.nn.functional.one_hot(coreset_target, num_classes=10 if "mnist" in args.dataset else 100)
	coreset_target = coreset_target.to(DEVICE).float()

	test_loss = AverageMeter()
	correct   = AverageMeter()

	with torch.no_grad():
		for data, target in loader:
			data = data.to(DEVICE)
			target = target.to(DEVICE)
			if task_id:
				features_test,   p_mu, p_log_var = net(data, task_id, post = False)
				features_coreset, r_mu, r_log_var  = net(coreset_input, task_id, post = True)
			else:
				features_test,   p_mu, p_log_var = net(data,          post = False)
				features_coreset, r_mu, r_log_var  = net(coreset_input, post = True)

			# compute_bases
			rs = sample(r_mu, r_log_var, args.d_rn_f, DEVICE).squeeze()

			x_supp_phi_t = rand_features(rs, torch.transpose(features_coreset, 1, 0), bias)  
			
			
			support_kernel = dotp_kernel(torch.transpose(x_supp_phi_t, 1, 0), x_supp_phi_t)  

			# closed-form solution with trainable lmd
			alpha = torch.matmul(torch.inverse(support_kernel + (torch.abs(lmd[task_id-1]) + 0.01) * torch.eye(support_kernel.shape[0]).to(DEVICE)), coreset_target)
			x_que_phi_t = rand_features(rs, torch.transpose(features_test, 1, 0), bias)

			# Compute cross kernel
			cross_kernel = dotp_kernel(torch.transpose(x_supp_phi_t, 1, 0), x_que_phi_t)

			pred = gma[task_id-1] * torch.matmul(cross_kernel.T, alpha) + bta[task_id-1]
			# align loss
			target_kernel = dotp_kernel(coreset_target, coreset_target.T)
			target_kernel = 0.99 * (target_kernel + 0.01)
			
			kernel_align_loss = cosine_dist(target_kernel, support_kernel)

			kl_loss = kl_div(r_mu, r_log_var, p_mu, p_log_var)
			
			loss = criterion(pred, target) + args.zeta * kernel_align_loss + args.tau * kl_loss
			
			test_loss.update(loss.item())

			acc = (pred.argmax(dim=-1) == target).float().mean()
			correct.update(acc.item())
	
	return {'accuracy': correct.avg, 'loss': test_loss.avg}

def get_benchmark_dataset(args):
	"""
	Returns the benchmark loader which could be either of these:
	get_split_cifar100_tasks, get_permuted_mnist_tasks, or get_rotated_mnist_tasks
	
	:param args:
	:return: a function which when called, returns all tasks
	"""
	if args.dataset == 'perm-mnist' or args.dataset == 'permuted-mnist':
		return get_permuted_mnist_tasks
	elif args.dataset == 'rot-mnist' or args.dataset == 'rotation-mnist':
		return get_rotated_mnist_tasks
	elif args.dataset == 'cifar-100' or args.dataset == 'cifar100':
		return get_split_cifar100_tasks
	elif args.dataset == 'mini-imagenet':
		return get_mini_imagenet_tasks
	else:
		raise Exception("Unknown dataset.\n"+
						"The code supports 'perm-mnist, rot-mnist, and cifar-100.")


def get_benchmark_model(args):
	"""
	Return the corresponding PyTorch model for experiment
	:param args:
	:return:
	"""
	if 'mnist' in args.dataset:
		if args.tasks == 20 and args.hiddens < 256:
			print("Warning! the main paper MLP with 256 neurons for experiment with 20 tasks")
		return MLP(args.hiddens, {'dropout': args.dropout}).to(DEVICE)
	elif 'cifar' in args.dataset:
		return ResNet18(config={'dropout': args.dropout}).to(DEVICE)
	elif 'imagenet' in args.dataset:
		return ResNet18MiniImagenet(config={'dropout': args.dropout}).to(DEVICE)
	else:
		raise Exception("Unknown dataset.\n"+
						"The code supports 'perm-mnist, rot-mnist, and cifar-100.")


def run(args):
	"""
	Run a single run of experiment.
	
	:param args: please see `utils.py` for arguments and options
	"""
	# pdb.set_trace()
	acc_db, loss_db, hessian_eig_db = init_experiment(args)
	print("Loading {} tasks for {}".format(args.tasks, args.dataset))
	tasks = get_benchmark_dataset(args)(args.tasks)
	print("loaded all tasks!")
	model = get_benchmark_model(args)

	print(f"Number of paramters: {count_parameters(model)}")

	# criterion
	criterion = nn.CrossEntropyLoss().to(DEVICE)
	bias = 2 * math.pi * torch.rand(args.d_rn_f, 1).to(DEVICE)
	time = 0

	# Learnable coefficient

	lmd   = torch.tensor([args.lmd for _ in range(args.tasks)], requires_grad=True, device=DEVICE)
	gma   = torch.tensor([1.0      for _ in range(args.tasks)], requires_grad=True, device=DEVICE)
	bta   = torch.tensor([0.       for _ in range(args.tasks)], requires_grad=True, device=DEVICE)

	coreset_all = []
	training_time = AverageMeter()
	# basis = []
	# rff = []
	# mu = []
	# log_var = []
	for current_task_id in range(1, args.tasks+1):
		print("================== TASK {} / {} =================".format(current_task_id, args.tasks))
		train_dataset = tasks[current_task_id]['train']
		train_dataset, coreset_dataset = coreset_selection(train_dataset, args.core_size, args.dataset)

		# import pdb; pdb.set_trace()
		# save coreset
		coreset_all.append(coreset_dataset)
		
		# uncomment for replay
		# coreset_dataset = torch.utils.data.dataset.ConcatDataset(coreset_all)
		# print(len(coreset_dataset))

		train_loader    = get_train_loader(train_dataset, args.batch_size)
		coreset_loader  = get_train_loader(coreset_dataset, args.batch_size)
		lr = max(args.lr * args.gamma ** (current_task_id), 0.00005) 
		for epoch in range(1, args.epochs_per_task+1):
			# 1. train and save
			params = list(model.parameters()) + [lmd, gma, bta]
			optimizer = torch.optim.SGD(params, lr=lr, momentum=0.8)  
			tic = benchmarker.time()
			train_single_epoch(model, optimizer, train_loader, coreset_loader, criterion, bias, lmd, gma, bta, args, current_task_id)
			toc = benchmarker.time()
			training_time.update(toc - tic)
			time += 1
			# import pdb; pdb.set_trace()
			# 2. evaluate on all tasks up to now, including the current task
			for prev_task_id in range(1, current_task_id+1):
				# 2.0. only evaluate once a task is finished
				if epoch == args.epochs_per_task:
					model = model.to(DEVICE)
					val_dataset = tasks[prev_task_id]['test']
					coreset_dataset = coreset_all[prev_task_id-1]

					val_loader      = get_test_loader(val_dataset)
					coreset_loader  = get_test_loader(coreset_dataset)
					
					# 2.1. compute accuracy and loss
					metrics = eval_single_epoch(model, val_loader, coreset_loader, criterion, bias, lmd, gma, bta, args, current_task_id)
					acc_db, loss_db = log_metrics(metrics, time, prev_task_id, current_task_id, acc_db, loss_db)
					
					# 2.2. (optional) compute eigenvalues and eigenvectors of Loss Hessian
					if prev_task_id == current_task_id and args.compute_eigenspectrum:
						hessian_eig_db = log_hessian(model, val_loader, time, prev_task_id, hessian_eig_db)
						
					# 2.3. save model parameters
					save_checkpoint(model, time)
		# basis, rff, mu, log_var = save_stats_one_task(model, coreset_all, bias, args, 1, basis, rff, mu, log_var)
	end_experiment(args, acc_db, loss_db, hessian_eig_db)
	global time_complexity
	print("Time Compexity of model for a given input samples", time_complexity.avg)
	print("Average Training Time", training_time.avg)


#%%




