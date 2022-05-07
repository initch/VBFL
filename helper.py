import torch
import config
import copy

def get_adversarial_index(id):
	i = 0
	for device in config.adversarial_list:
		if device == id:
			break
		i += 1

	return i % 4

def get_poison_batch(images, targets, adversarial_index=-1, evaluation=False):

	poison_count = 0
	new_images = images.clone().detach()
	new_targets = targets.clone().detach()

	for index in range(0, len(images)):
		if evaluation:  # poison all data when testing
			new_targets[index] = config.poison_label_swap
			new_images[index] = add_pixel_pattern(images[index], adversarial_index)
			poison_count += 1

		else:  # poison part of data when training
			if index < config.poisoning_per_batch:
				new_targets[index] = config.poison_label_swap
				new_images[index] = add_pixel_pattern(images[index], adversarial_index)
				poison_count += 1
			else:
				new_images[index] = images[index]
				new_targets[index] = targets[index]

	if evaluation:
		new_images.requires_grad_(False)
		new_targets.requires_grad_(False)
		
	return new_images, new_targets, poison_count


def add_pixel_pattern(ori_image, adversarial_index):
	image = copy.deepcopy(ori_image)

	poison_patterns = []
	if adversarial_index == -1:
		for i in range(0, config.trigger_num):
			poison_patterns = poison_patterns + config.params[str(i) + '_poison_pattern']
	else:
		poison_patterns = config.params[str(adversarial_index) + '_poison_pattern']

	for i in range(0, len(poison_patterns)):
		pos = poison_patterns[i]
		index = 28 * pos[0] + pos[1]
		image[index] = 1

	return image


def poison_test_dataset(test_dataset):
	'''delete the test data with target label'''
	test_classes = {}
	for ind, x in enumerate(test_dataset):
		_, label = x
		label = int(label)
		if label in test_classes:
			test_classes[label].append(ind)
		else:
			test_classes[label] = [ind]

	range_no_id = list(range(0, len(test_dataset)))
	for image_ind in test_classes[config.poison_label_swap]:
		if image_ind in range_no_id:
			range_no_id.remove(image_ind)
	poison_label_inds = test_classes[config.poison_label_swap]

	return torch.utils.data.DataLoader(test_dataset,
						batch_size=config.test_batch_size,
						sampler=torch.utils.data.sampler.SubsetRandomSampler(
							range_no_id))


def test_poison_trigger(model, dev, test_ds, adv_index=-1):
	'''Calculating attack sccuess rate for the specified trigger'''
	with torch.no_grad():
		total_loss = 0.0
		correct = 0
		dataset_size = 0
		poison_data_count = 0

		poison_test_dl = helper.poison_test_dataset(test_ds)

		for data, targets in data_iterator:
			data, targets, poison_num = get_poison_batch(data, targets, adversarial_index=adv_index, evaluation=True)
			data, targets = data.to(dev), targets.to(dev)
			poison_data_count += poison_num
			dataset_size += len(data)
			output = model(data)
			total_loss += torch.nn.functional.cross_entropy(output, targets,reduction='sum').item()  # sum up batch loss
			pred = output.data.max(1)[1]  # get the index of the max log-probability
			correct += pred.eq(targets.data.view_as(pred)).cpu().sum().item()

		acc = (float(correct) / float(poison_data_count)) if poison_data_count!=0 else 0
		total_l = total_loss / poison_data_count if poison_data_count!=0 else 0

		# return total_l, acc, correct, poison_data_count
		return acc

