from plot_helper import *

dict = {
    '4/20': '04062022_232742',
    '8/20': '04072022_152250',
    '12/20': '04072022_221816',
    '16/20': '04082022_111128'
}

plot_acc_for_trails('acc_malicious_ratio', dict, 60, 1, 4, share=True)
plot_asr_for_trails('asr_malicious_ratio', dict, 60, 2, 2)
plot_acc_asr_for_trails('malicious_ratio', dict, 15, 50)