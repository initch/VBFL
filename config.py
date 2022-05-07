lr = 1e-5
poison_lr = 1e-5
decay = 1e-8

num_devices = 10
roles_assign = [5, 2, 3]
adversarial_list = [0,1,2,3]

global_update_method = 'all_devices'
#global_update_method = 'only_workers'

# roles_assign_method
# 1 - assign randomly
# 2 - static contribution, assign according to probability
# 3 - TODO: dynamic contribution, assign according to ranks
roles_assign_method = 2

attacking_shot = 200
start_poison_round = 10
start_detection_round = 200
poison_label_swap = 2
batch_size = 64
poisoning_per_batch = 64
test_batch_size = 100
trigger_num = 4
data_assign = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1] #, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
local_epochs = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1] #, 1, 1, 1, 1, 1, 1, 1, 1, 1, 10]

params = {
	## gap 2 size 1*4 base (0, 0)
	'0_poison_pattern': [[3, 5], [3, 6], [3, 7], [3, 8]],
	'1_poison_pattern': [[0, 5], [0, 6], [0, 7], [0, 8]],
	'2_poison_pattern': [[3, 0], [3, 1], [3, 2], [3, 3]],
	'3_poison_pattern': [[0, 0], [0, 1], [0, 2], [0, 3]]
}

	#'0_poison_pattern': [[0, 0], [0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 7], [0, 8]],
	#'1_poison_pattern': [[0, 10], [0, 11], [0, 12], [0,13], [0,14], [0,15], [0,16],[0,17], [0,18]],
	#'2_poison_pattern': [[3, 0], [3, 1], [3, 2], [3, 3], [3, 4], [3, 5], [3,6], [3,7], [3,8]],
	#'3_poison_pattern': [[3, 10], [3, 11], [3, 12], [3,13], [3,14], [3,15], [3,16],[3,17], [3,18]],