from plot_helper import *

# poison 24/64
# malicious 4/10

dict = {
    'Size 1*3': "04052022_213757",
    #'Size 2*2': "04062022_113413",
    #'Size 2*3': "04062022_003444",
    'Size 1*6': "04062022_154815",
    'Size 1*9': "04062022_183103"
}

plot_acc_for_trails('acc_trigger_size', dict, 170, 1, 3,share=True)
plot_asr_for_trails('asr_trigger_size', dict, 170, 1, 3)