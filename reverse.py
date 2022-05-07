import torch
import numpy as np
from torch.nn import CrossEntropyLoss
import tqdm
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from DatasetLoad import DatasetLoad


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def train(model, target_label, train_loader, param):
	print("Processing label: {}".format(target_label))

	width, height = param["image_size"]
	trigger = torch.rand((width, height), requires_grad=True)
	trigger = trigger.to(device).detach().requires_grad_(True)
	mask = torch.rand((width, height), requires_grad=True)
	mask = mask.to(device).detach().requires_grad_(True)

	Epochs = param["Epochs"]
	lamda = param["lamda"]

	min_norm = np.inf
	min_norm_count = 0

	criterion = CrossEntropyLoss()
	optimizer = torch.optim.Adam([{"params": trigger},{"params": mask}],lr=0.005)
	model.to(device)
	model.eval()

	for epoch in range(Epochs):
		norm = 0.0
		for images, _ in tqdm.tqdm(train_loader, desc='Epoch %3d' % (epoch + 1)):
			optimizer.zero_grad()
			images = images.reshape(len(images), 28, 28)
			images = images.to(device)
			trojan_images = (1 - torch.unsqueeze(mask, dim=0)) * images + torch.unsqueeze(mask, dim=0) * trigger
			y_pred = model(trojan_images)
			y_target = torch.full((y_pred.size(0),), target_label, dtype=torch.long).to(device)
			loss = criterion(y_pred, y_target) + lamda * torch.sum(torch.abs(mask))
			loss.backward()
			optimizer.step()

			# figure norm
			with torch.no_grad():
				# 防止trigger和norm越界
				torch.clip_(trigger, 0, 1)
				torch.clip_(mask, 0, 1)
				norm = torch.sum(torch.abs(mask))
		print("norm: {}".format(norm))

		# to early stop
		if norm < min_norm:
			min_norm = norm
			min_norm_count = 0
		else:
			min_norm_count += 1

		if min_norm_count > 30:
			break

	return trigger.cpu(), mask.cpu()



def reverse_engineer(model, weights_to_eval, train_loader):
	param = {
		"dataset": "mnist",
		"Epochs": 5,
		"batch_size": 64,
		"lamda": 0.01,
		"num_classes": 10,
		"image_size": (28, 28)
	}

	# test
	'''model = torch.load("model_device_1_comm_3.pkl")

	mnist_dataset = DatasetLoad('mnist', 1)
	test_data = torch.tensor(mnist_dataset.test_data)
	test_label = torch.argmax(torch.tensor(mnist_dataset.test_label), dim=1)
	test_data = TensorDataset(test_data, test_label)

	train_loader = DataLoader(test_data, batch_size=param["batch_size"], shuffle=False)'''

	model.load_state_dict(weights_to_eval, strict=True)
	norm_list = []
	masks = []
	triggers = []
	for label in range(param["num_classes"]):
		trigger, mask = train(model, label, train_loader, param)
		norm_list.append(mask.sum().item())
		mask = mask.cpu().detach().numpy()
		trigger = trigger.cpu().detach().numpy()
		masks.append(mask)
		triggers.append(trigger)

	return masks

def plot_masks(masks, name):
	plt.clf()
	fig, axs = plt.subplots(2, 5)
	for cnt in range(10):
		row = int(cnt/ 5)
		col = cnt % 5
		axs[row, col].set_xticks([])
		axs[row, col].set_yticks([])
		axs[row, col].set_title(f"{cnt}")
		axs[row, col].imshow(masks[cnt], cmap='binary')
		
	plt.tight_layout()
	plt.savefig(f'images/{name}.png')


def outlier_detection(l1_norm_list):

	consistency_constant = 1.4826  # if normal distribution
	median = np.median(l1_norm_list)
	mad = consistency_constant * np.median(np.abs(l1_norm_list - median))
	min_mad = np.abs(np.min(l1_norm_list) - median) / mad

	print('median: %f, MAD: %f' % (median, mad))
	print('anomaly index: %f' % min_mad)

	flag_list = []
	for y_label in range(10):
		if l1_norm_list[y_label] > median:
			continue
		if np.abs(l1_norm_list[y_label] - median) / mad > 2:
			flag_list.append((y_label, l1_norm_list[y_label]))

	if len(flag_list) > 0:
		flag_list = sorted(flag_list, key=lambda x: x[1])

	print('flagged label list: %s' %
		  ', '.join(['%d: %2f' % (y_label, l_norm)
					 for y_label, l_norm in flag_list]))

	return flag_list


def analyze_pattern_norm_dist(masks):

	mask_flatten = []

	for y_label in range(10):
		mask = masks[y_label]
		mask_flatten.append(mask.flatten())

	l1_norm_list = [np.sum(np.abs(m)) for m in mask_flatten]

	print('%d labels found' % len(l1_norm_list))

	flag_list = outlier_detection(l1_norm_list)

	return flag_list

if __name__ == "__main__":
	masks = reverse_engineer()
	analyze_pattern_norm_dist(masks)