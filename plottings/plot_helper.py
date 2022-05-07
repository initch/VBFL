from matplotlib import pyplot as plt
from os import listdir

colors = ['lightsteelblue', 'cornflowerblue', 'lightsalmon', 'peachpuff']

def plot_acc_for_single_trial(time):

    log_folder = f"logs/{time}"
    all_rounds_log_files = sorted([f for f in listdir(log_folder) if f.startswith('comm')], key=lambda x: int(x.split('.')[0].split('_')[-1]))
    rounds = len(all_rounds_log_files) - 1
    accu = []

    for i in range(rounds):
        
        with open(f"{log_folder}/comm_{i+1}/accuracy_comm_{i+1}.txt", 'r') as file:
            lines_list = file.read().split("\n")
            accu.append(float(lines_list[0].split(': ')[-1]))

    plt.clf()
    plt.xlabel('Communication Round')
    plt.ylabel('Accuracy')

    plt.plot(range(1, rounds+1), accu)

    plt.savefig(f"{log_folder}/1_accuracy.png", dpi=300)
    plt.show()


def plot_asr_for_single_trail(time):
    log_folder = f"logs/{time}"
    all_rounds_log_files = sorted([f for f in listdir(log_folder) if f.startswith('comm')], key=lambda x: int(x.split('.')[0].split('_')[-1]))
    rounds = len(all_rounds_log_files) - 1
    global_trigger = []
    local_trigger_1 = []
    local_trigger_2 = []
    local_trigger_3 = []
    local_trigger_4 = []

    for i in range(rounds):
        with open(f"{log_folder}/comm_{i+1}/attack_comm_{i+1}.txt", 'r') as file:
            lines_list = file.read().split("\n")
            global_trigger.append(float(lines_list[0].split(': ')[-1]))
            local_trigger_1.append(float(lines_list[1].split(': ')[-1]))
            local_trigger_2.append(float(lines_list[2].split(': ')[-1]))
            local_trigger_3.append(float(lines_list[3].split(': ')[-1]))
            local_trigger_4.append(float(lines_list[4].split(': ')[-1]))

    plt.clf()
    plt.xlabel('Communication Round')
    plt.ylabel('Attack Success Rate')

    plt.plot(range(1, rounds+1), local_trigger_1, color='lightsteelblue', label='Local trigger 1')
    plt.plot(range(1, rounds+1), local_trigger_2, color='cornflowerblue', label='Local trigger 2')
    plt.plot(range(1, rounds+1), local_trigger_3, color='lightsalmon', label='Local trigger 3')
    plt.plot(range(1, rounds+1), local_trigger_4, color='peachpuff', label='Local trigger 4')
    plt.plot(range(1, rounds+1), global_trigger, linewidth=0.8, color='black', label='Global trigger')

    plt.legend(loc='best')

    plt.savefig(f"{log_folder}/2_attack.png", dpi=300)
    plt.show()

def plot_acc_for_single_device(time, id):
    log_folder = f"logs/{time}"
    all_rounds_log_files = sorted([f for f in listdir(log_folder) if f.startswith('comm')], key=lambda x: int(x.split('.')[0].split('_')[-1]))
    rounds = len(all_rounds_log_files) - 1
    accu = []

    for i in range(rounds):
        with open(f"{log_folder}/comm_{i+1}/accuracy_comm_{i+1}.txt", 'r') as file:
            lines_list = file.read().split("\n")
            for line in lines_list:
                if f'device_{id}' in line:
                    accu.append(line.split(': ')[-1])
                    break

    plt.clf()
    plt.xlabel('Communication Round')
    plt.ylabel('Accuracy')

    plt.plot(range(1, rounds+1), accu)

    plt.legend(loc='best')

    plt.savefig(f"{log_folder}/device_{id}_acc.png", dpi=300)
    plt.show()

def plot_asr_for_single_device(time, id):
    log_folder = f"logs/{time}"
    all_rounds_log_files = sorted([f for f in listdir(log_folder) if f.startswith('comm')], key=lambda x: int(x.split('.')[0].split('_')[-1]))
    rounds = len(all_rounds_log_files) - 1
    global_trigger = []
    local_trigger_1 = []
    local_trigger_2 = []
    local_trigger_3 = []
    local_trigger_4 = []

    for i in range(rounds):
        with open(f"{log_folder}/comm_{i+1}/device_{id}_asr_comm_{i+1}.txt", 'r') as file:
            lines_list = file.read().split("\n")
            global_trigger.append(float(lines_list[2].split(': ')[-1]))
            local_trigger_1.append(float(lines_list[3].split(': ')[-1]))
            local_trigger_2.append(float(lines_list[4].split(': ')[-1]))
            local_trigger_3.append(float(lines_list[5].split(': ')[-1]))
            local_trigger_4.append(float(lines_list[6].split(': ')[-1]))

    plt.clf()
    plt.xlabel('Communication Round')
    plt.ylabel('Attack Success Rate')

    plt.plot(range(1, rounds+1), local_trigger_1, color='lightsteelblue', label='Local trigger 1')
    plt.plot(range(1, rounds+1), local_trigger_2, color='cornflowerblue', label='Local trigger 2')
    plt.plot(range(1, rounds+1), local_trigger_3, color='lightsalmon', label='Local trigger 3')
    plt.plot(range(1, rounds+1), local_trigger_4, color='peachpuff', label='Local trigger 4')
    plt.plot(range(1, rounds+1), global_trigger, linewidth=0.8, color='black', label='Global trigger')

    plt.legend(loc='best')

    plt.savefig(f"{log_folder}/device_{id}_asr.png", dpi=300)
    plt.show()

def plot_acc_for_all_devices(time, num_devices, t_row, t_col):
    log_folder = f"logs/{time}"
    all_rounds_log_files = sorted([f for f in listdir(log_folder) if f.startswith('comm')], key=lambda x: int(x.split('.')[0].split('_')[-1]))
    rounds = len(all_rounds_log_files) - 1
    fig, axs = plt.subplots(t_row, t_col)
    fig.set_figwidth(15)
    cnt = 0
    for id in range(1, num_devices+1):
        accu = []
        for i in range(rounds):
            with open(f"{log_folder}/comm_{i+1}/accuracy_comm_{i+1}.txt", 'r') as file:
                lines_list = file.read().split("\n")
                for line in lines_list:
                    if f'device_{id}' in line:
                        accu.append(float(line.split(': ')[-1]))
                        break

        if t_row > 1: # 2-dimentional
            row = int(cnt/ t_col)
            col = cnt % t_col
            axs[row, col].set_ylim([0,1.0])
            axs[row, col].plot(range(1, rounds+1), accu)
            axs[row, col].set_title(f"device_{id}")
            axs[row, col].set_xlabel("Communication Rounds")
            axs[row, col].set_ylabel("Accuracy")
        else:
            axs[cnt].set_ylim([0,1.0])
            axs[cnt].plot(range(1, rounds+1), accu)
            axs[cnt].set_title(f"device_{id}")
            axs[cnt].set_xlabel("Communication Rounds")
            axs[cnt].set_ylabel("Accuracy")

        cnt += 1
    
    plt.tight_layout()
    plt.savefig(f"{log_folder}/all_devices_acc.png", dpi=300)
    plt.show()


def plot_asr_for_all_devices(time, num_devices, t_row, t_col):
    log_folder = f"logs/{time}"
    all_rounds_log_files = sorted([f for f in listdir(log_folder) if f.startswith('comm')], key=lambda x: int(x.split('.')[0].split('_')[-1])) 
    rounds = len(all_rounds_log_files) - 1
    fig, axs = plt.subplots(t_row, t_col, figsize=(6,6))
    cnt = 0
    fc = {1:1, 3:2, 5:3, 7:4, 9:5, 10:6}
    for id, value in fc.items():
        global_trigger = []
        local_trigger_1 = []
        local_trigger_2 = []
        local_trigger_3 = []
        local_trigger_4 = []
        for i in range(rounds):
            with open(f"{log_folder}/comm_{i+1}/device_{id}_asr_comm_{i+1}.txt", 'r') as file:
                lines_list = file.read().split("\n")
                global_trigger.append(float(lines_list[2].split(': ')[-1]))
                local_trigger_1.append(float(lines_list[3].split(': ')[-1]))
                local_trigger_2.append(float(lines_list[4].split(': ')[-1]))
                local_trigger_3.append(float(lines_list[5].split(': ')[-1]))
                local_trigger_4.append(float(lines_list[6].split(': ')[-1]))

        if t_row > 1: # 2-dimentional
            row = int(cnt/ t_col)
            col = cnt % t_col
            axs[row, col].set_ylim([0,1.0])
            axs[row, col].plot(range(1, rounds+1), local_trigger_1, color='lightsteelblue', label='Local trigger 1')
            axs[row, col].plot(range(1, rounds+1), local_trigger_2, color='cornflowerblue', label='Local trigger 2')
            axs[row, col].plot(range(1, rounds+1), local_trigger_3, color='lightsalmon', label='Local trigger 3')
            axs[row, col].plot(range(1, rounds+1), local_trigger_4, color='peachpuff', label='Local trigger 4')
            axs[row, col].plot(range(1, rounds+1), global_trigger, linewidth=1, color='black', label='Global trigger')
            axs[row, col].legend(fontsize=5)
            axs[row, col].set_title(f"SL_node_{value}")
            axs[row, col].set_ylim([0,1.1])
            axs[row, col].set_xlabel("Communication Rounds")
            axs[row, col].set_ylabel("Attack Success Rate")
        else:
            axs[cnt].set_ylim([0,1.0])
            axs[cnt].plot(range(1, rounds+1), local_trigger_1, color='lightsteelblue', label='Local trigger 1')
            axs[cnt].plot(range(1, rounds+1), local_trigger_2, color='cornflowerblue', label='Local trigger 2')
            axs[cnt].plot(range(1, rounds+1), local_trigger_3, color='lightsalmon', label='Local trigger 3')
            axs[cnt].plot(range(1, rounds+1), local_trigger_4, color='peachpuff', label='Local trigger 4')
            axs[cnt].plot(range(1, rounds+1), global_trigger, linewidth=1, color='black', label='Global trigger')
            
            axs[cnt].set_title(f"device_{id}")
            axs[cnt].set_xlabel("Communication Rounds")
            axs[cnt].set_ylabel("Attack Success Rate")

        cnt += 1
    
    lines, labels = fig.axes[-1].get_legend_handles_labels()
    plt.tight_layout()
    plt.savefig(f"{log_folder}/all_devices_asr.png", dpi=300)
    plt.show()

def plot_workers(time, adv_index, num_devices):
    log_folder = f"logs/{time}"
    all_rounds_log_files = sorted([f for f in listdir(log_folder) if f.startswith('comm')], key=lambda x: int(x.split('.')[0].split('_')[-1])) 
    rounds = len(all_rounds_log_files) - 1
    plt.clf()
    plt.xlabel("Communication Rounds")
    plt.ylabel("Acc / ASR")
    acc = []
    asr = []
    glo = []
    for i in range(1, rounds+1):
        with open(f"{log_folder}/comm_{i}/accuracy_comm_{i}.txt", 'r') as file:
            lines_list = file.read().split("\n")
            for line in lines_list:
                if f'worker' in line:
                    acc.append(float(line.split(': ')[-1]))
                    break
        for id in range(1, num_devices+1):
            with open(f"{log_folder}/comm_{i}/device_{id}_asr_comm_{i}.txt", 'r') as file:
                lines_list = file.read().split("\n")
                if lines_list[1].split(': ')[-1] == 'worker':
                    if adv_index == -1: #global trigger
                        glo.append(float(lines_list[2].split(': ')[-1]))
                    else: #local trigger
                        asr.append(float(lines_list[3+adv_index].split(': ')[-1]))
                        glo.append(float(lines_list[2].split(': ')[-1]))
                    break
    plt.plot(range(1,rounds+1), acc, color=colors[1], label='Acc-workers')
    #plt.plot(range(1,rounds+1), asr, marker="o",color=colors[3], label=f'ASR-trigger-{adv_index+1}')
    plt.plot(range(1,rounds+1), glo, color=colors[2], label='ASR-workers')
    plt.legend()
    plt.savefig(f"{log_folder}/workers_trigger_{adv_index+1}.png", dpi=300)
    plt.show()

def plot_asr_for_single_trigger(time, adv_index, num_devices, t_row, t_col):
    log_folder = f"logs/{time}"
    all_rounds_log_files = sorted([f for f in listdir(log_folder) if f.startswith('comm')], key=lambda x: int(x.split('.')[0].split('_')[-1])) 
    rounds = len(all_rounds_log_files) - 1
    fig, axs = plt.subplots(t_row, t_col)
    fig.set_figwidth(15)
    cnt = 0
    for id in range(1, num_devices+1):
        w_r = []
        non_w_r = []
        worker = []
        non_worker = []
        for i in range(1, rounds+1):
            with open(f"{log_folder}/comm_{i}/device_{id}_asr_comm_{i}.txt", 'r') as file:
                lines_list = file.read().split("\n")
                if lines_list[1].split(': ')[-1] == 'worker':
                    w_r.append(i)
                    if adv_index == -1: #global trigger
                        worker.append(float(lines_list[2].split(': ')[-1]))
                    else: #local trigger
                        worker.append(float(lines_list[3+adv_index].split(': ')[-1]))
                else:
                    non_w_r.append(i)
                    if adv_index == -1:
                        non_worker.append(float(lines_list[2].split(': ')[-1]))
                    else:
                        non_worker.append(float(lines_list[3+adv_index].split(': ')[-1]))

        if t_row > 1: # 2-dimentional
            row = int(cnt/ t_col)
            col = cnt % t_col
            axs[row, col].set_ylim([0,1.0])
            axs[row, col].scatter(non_w_r, non_worker, marker="*",color='lightsteelblue')
            axs[row, col].scatter(w_r, worker, marker="o",color='lightsalmon')
            axs[row, col].set_title(f"device_{id}")
            axs[row, col].set_xlabel("Communication Rounds")
            axs[row, col].set_ylabel("Attack Success Rate")
        else:
            axs[cnt].set_ylim([0,1.0])
            axs[cnt].scatter(non_w_r, non_worker, marker="*",color='lightsteelblue')
            axs[cnt].scatter(w_r, worker, marker="o",color='lightsalmon')
            axs[cnt].set_title(f"device_{id}")
            axs[cnt].set_xlabel("Communication Rounds")
            axs[cnt].set_ylabel("Attack Success Rate")

        cnt += 1
    
    plt.tight_layout()
    plt.savefig(f"{log_folder}/asr_trigger_{adv_index+1}.png", dpi=300)
    plt.show()

def plot_acc_for_trails(name, log_dict, rounds, t_row, t_col, share=False):
    plt.clf()
    if not share:
        fig, axs = plt.subplots(t_row, t_col, figsize=(14,6))
    else:
        plt.xlabel('Communication Round')
        plt.ylabel('Accuracy')
    data = {}
    cnt = 0
    color_index = 0
    for var, time in log_dict.items():
        log_folder = f"logs/{time}"
        all_rounds_log_files = sorted([f for f in listdir(log_folder) if f.startswith('comm')], key=lambda x: int(x.split('.')[0].split('_')[-1]))
        total_rounds = len(all_rounds_log_files) - 1
        if total_rounds < rounds:
            print(f"[Error] logs/{time} has saved only {total_rounds} rounds.\n")
            return
        accu = []
        for i in range(rounds):
            with open(f"{log_folder}/comm_{i+1}/accuracy_comm_{i+1}.txt", 'r') as file:
                lines_list = file.read().split("\n")
                accu.append(float(lines_list[0].split(': ')[-1]))
        data[var] = accu

        if not share:
            if t_row > 1:
                row = int(cnt/ t_col)
                col = cnt % t_col
                axs[row, col].plot(range(1, rounds+1), data[var])
                axs[row, col].set_title(var)
                axs[row, col].set_xlabel("Communication Rounds")
                axs[row, col].set_ylabel("Accuracy")
            else:
                axs[cnt].plot(range(1, rounds+1), data[var])
                axs[cnt].set_title(var)
                axs[cnt].set_xlabel("Communication Rounds")
                axs[cnt].set_ylabel("Accuracy")
        
        else:
            plt.plot(range(1, rounds+1), data[var],color=colors[color_index], label=var)
            plt.yticks([0,0.2,0.4,0.6,0.8,1.0])
            color_index += 1
            
        cnt += 1

    plt.tight_layout()
    plt.legend(loc='lower right')
    
    plt.savefig(f"images/{name}.png", dpi=300)
    plt.show()
    return

def plot_asr_for_trails(name, log_dict, rounds, t_row, t_col):
    plt.clf()
    fig, axs = plt.subplots(t_row, t_col, figsize=(14,6))
    data = {}
    cnt = 0
    for var, time in log_dict.items():
        log_folder = f"logs/{time}"
        all_rounds_log_files = sorted([f for f in listdir(log_folder) if f.startswith('comm')], key=lambda x: int(x.split('.')[0].split('_')[-1]))
        total_rounds = len(all_rounds_log_files) - 1
        if total_rounds < rounds:
            print(f"[Error] logs/{time} has saved only {total_rounds} rounds.\n")
            return
        else:
            global_trigger = []
            local_trigger_1 = []
            local_trigger_2 = []
            local_trigger_3 = []
            local_trigger_4 = []

            for i in range(rounds):
                with open(f"{log_folder}/comm_{i+1}/attack_comm_{i+1}.txt", 'r') as file:
                    lines_list = file.read().split("\n")
                    global_trigger.append(float(lines_list[0].split(': ')[-1]))
                    local_trigger_1.append(float(lines_list[1].split(': ')[-1]))
                    local_trigger_2.append(float(lines_list[2].split(': ')[-1]))
                    local_trigger_3.append(float(lines_list[3].split(': ')[-1]))
                    local_trigger_4.append(float(lines_list[4].split(': ')[-1]))

            if t_row > 1: # 2-dimentional
                row = int(cnt/ t_col)
                col = cnt % t_col
                axs[row, col].plot(range(1, rounds+1), local_trigger_1, color='lightsteelblue', label='Local trigger 1')
                axs[row, col].plot(range(1, rounds+1), local_trigger_2, color='cornflowerblue', label='Local trigger 2')
                axs[row, col].plot(range(1, rounds+1), local_trigger_3, color='lightsalmon', label='Local trigger 3')
                axs[row, col].plot(range(1, rounds+1), local_trigger_4, color='peachpuff', label='Local trigger 4')
                axs[row, col].plot(range(1, rounds+1), global_trigger, linewidth=1, color='black', label='Global trigger')
                
                axs[row, col].set_title(var)
                axs[row, col].set_xlabel("Communication Rounds")
                axs[row, col].set_ylabel("Attack Success Rate")
                axs[row, col].legend(loc='lower right', fontsize=4)
            else:
                axs[cnt].plot(range(1, rounds+1), local_trigger_1, color='lightsteelblue', label='Local trigger 1')
                axs[cnt].plot(range(1, rounds+1), local_trigger_2, color='cornflowerblue', label='Local trigger 2')
                axs[cnt].plot(range(1, rounds+1), local_trigger_3, color='lightsalmon', label='Local trigger 3')
                axs[cnt].plot(range(1, rounds+1), local_trigger_4, color='peachpuff', label='Local trigger 4')
                axs[cnt].plot(range(1, rounds+1), global_trigger, linewidth=1, color='black', label='Global trigger')
                
                axs[cnt].set_title(f"{var}")
                axs[cnt].set_xlabel("Communication Rounds")
                axs[cnt].set_ylabel("Attack Success Rate")
                axs[cnt].legend(loc='lower right', fontsize=6)

            cnt += 1
    
    plt.tight_layout()
    plt.savefig(f"images/{name}.png", dpi=300)
    plt.show()
    return

def plot_acc_asr_for_trails(name, dict, round_1, round_2):
    plt.clf()
    plt.xlabel('Communication Round')
    plt.ylabel('Acc / ASR')
    
    x = []
    acc_1 = []
    acc_2 = []
    asr_1 = []
    asr_2 = []

    for var, time in dict.items():
        log_folder = f"logs/{time}"
        all_rounds_log_files = sorted([f for f in listdir(log_folder) if f.startswith('comm')], key=lambda x: int(x.split('.')[0].split('_')[-1]))
        total_rounds = len(all_rounds_log_files) - 1
        if total_rounds < round_1:
            print(f"[Error] logs/{time} has saved only {total_rounds} rounds.\n")
            return
        x.append(int(var.split('/')[0]))
        with open(f"{log_folder}/comm_{round_1}/accuracy_comm_{round_1}.txt", 'r') as file:
            lines_list = file.read().split("\n")
            acc_1.append(float(lines_list[0].split(': ')[-1]))
        with open(f"{log_folder}/comm_{round_2}/accuracy_comm_{round_2}.txt", 'r') as file:
            lines_list = file.read().split("\n")
            acc_2.append(float(lines_list[0].split(': ')[-1]))
        with open(f"{log_folder}/comm_{round_1}/attack_comm_{round_1}.txt", 'r') as file:
            lines_list = file.read().split("\n")
            asr_1.append(float(lines_list[0].split(': ')[-1]))
        with open(f"{log_folder}/comm_{round_2}/attack_comm_{round_2}.txt", 'r') as file:
            lines_list = file.read().split("\n")
            asr_2.append(float(lines_list[0].split(': ')[-1]))
    
    plt.plot(x, acc_1, color='lightsteelblue', marker='s', label=f'Acc-{round_1}')
    plt.plot(x, acc_2, color='cornflowerblue', marker='s', label='Acc')
    plt.plot(x, asr_1, color='lightsalmon', marker='o', label=f'ASR-{round_1}')
    plt.plot(x, asr_2, color='peachpuff', marker='o', label='ASR')
    plt.yticks([0,0.2,0.4,0.6,0.8,1.0])

    plt.legend(loc='best')

    plt.savefig(f"images/{name}.png", dpi=300)
    plt.show()

def plot_single_device_for_trails(name, dict, device_id, rounds, trigger_index):
    plt.clf()
    plt.xlabel('Communication Round')
    plt.ylabel('Acc / ASR')
    
    x = []
    acc = []
    asr_1 = []
    asr_2 = []

    for var, time in dict.items():
        log_folder = f"logs/{time}"
        all_rounds_log_files = sorted([f for f in listdir(log_folder) if f.startswith('comm')], key=lambda x: int(x.split('.')[0].split('_')[-1]))
        total_rounds = len(all_rounds_log_files) - 1
        if total_rounds < rounds:
            print(f"[Error] logs/{time} has saved only {total_rounds} rounds.\n")
            return
        x.append(int(var))
        with open(f"{log_folder}/comm_{rounds}/accuracy_comm_{rounds}.txt", 'r') as file:
            lines_list = file.read().split("\n")
            for line in lines_list:
                if f"device_{device_id}" in line:
                    acc.append(float(line.split(': ')[-1]))
                    break
        with open(f"{log_folder}/comm_{rounds}/device_{device_id}_asr_comm_{rounds}.txt", 'r') as file:
            lines_list = file.read().split("\n")
            asr_1.append(float(lines_list[trigger_index+2].split(': ')[-1]))
        with open(f"{log_folder}/comm_{rounds}/device_{device_id}_asr_comm_{rounds}.txt", 'r') as file:
            lines_list = file.read().split("\n")
            asr_2.append(float(lines_list[2].split(': ')[-1]))
                
    
    plt.plot(x, acc, color='cornflowerblue', marker='s', label=f'Acc')
    plt.plot(x, asr_1, color='peachpuff', marker='o', label=f'ASR-local-trigger-{trigger_index}') # local
    plt.plot(x, asr_2, color='lightsalmon', marker='o', label='ASR-global-trigger') # global trigger

    plt.legend(loc='best')

    plt.savefig(f"images/{name}.png", dpi=300)
    plt.show()