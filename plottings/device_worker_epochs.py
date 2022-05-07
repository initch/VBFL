from plot_helper import *

# benign worker - local epochs
benign_dict = {
    '1': '04122022_144916',
    '2': '04162022_195253',
    '3': '04162022_224955',
    '4': '04172022_122819',
    '5': '04172022_155605',
    '6': '04172022_194523',
    '7': '04182022_092907'
}

plot_single_device_for_trails("propagation_local_epochs", dict, 8, 16, 1)