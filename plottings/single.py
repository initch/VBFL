from plot_helper import *
import sys

time = sys.argv[1]

# global_update_method only_workers 04102022_101651

plot_acc_for_single_trial(time)
plot_asr_for_single_trail(time)
#plot_asr_for_single_device(time, 2)

#plot_workers(time, -1, 6)
#plot_asr_for_single_device(time, 10)
#plot_acc_for_all_devices(time, 10, 2, 5)
#plot_asr_for_all_devices(time, 10, 3, 2)

#plot_asr_for_single_trigger(time, -1, 6, 2, 3)