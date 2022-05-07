from  Digraph import Digraph
import sys
from os import listdir

devices = 6
workers = 3
malicious_list = [0]
start_round = 10
ah = 0.1
m = ['B']*devices
for i in range(devices):
	if i in malicious_list:
		m[i] = 'M'

open("nodes.txt", 'w').close()
open("edges.txt", 'w').close()

time = sys.argv[1]

log_folder = f"logs/{time}"
all_rounds_log_files = sorted([f for f in listdir(log_folder) if f.startswith('comm')], key=lambda x: int(x.split('.')[0].split('_')[-1]))
rounds = len(all_rounds_log_files) - 1

def get_asr_last_round(d, i):
	with open(f"{log_folder}/comm_{i-1}/device_{d}_asr_comm_{i-1}.txt", 'r') as file:
		lines_list = file.read().split("\n")
		return float(lines_list[2].split(': ')[-1])

flag = [0]*devices
current_index = [0]*workers
cnt = 0

for i in range(start_round-1, 60):
	# find workers in this round
	workers_this_round = []
	asr_this_round = 0
	workers_cnt = 0
	for d in range(1, devices+1):
		with open(f"{log_folder}/comm_{i}/device_{d}_asr_comm_{i}.txt", 'r') as file:
			lines_list = file.read().split("\n")
			if i == start_round-1:
				cnt += 1
				flag[d-1] = cnt
				name = int(d)
				with open("nodes.txt", 'a') as f:
					f.write(f"{cnt} {name} {m[d-1]} {i} 0\n")
			if i >= start_round and lines_list[1].split(': ')[-1] == 'worker':
				cnt += 1
				current_index[workers_cnt] = cnt
				workers_cnt += 1
				name = int(d)
				workers_this_round.append(name)
				t = i
				asr = float(lines_list[2].split(': ')[-1])
				asr_this_round = asr
				with open("nodes.txt", 'a') as f:
					f.write(f"{cnt} {name} {m[d-1]} {t} {asr}\n")
	
	workers_cnt = 0
	for worker in workers_this_round:
		if asr_this_round > ah:
			asr_last_round = get_asr_last_round(worker, i)
			if asr_this_round > asr_last_round or asr_this_round == 1.0:
				# find edges
				for w in workers_this_round:
					if m[w-1] == 'M' or get_asr_last_round(w, i) > ah:
						with open("edges.txt", 'a') as f:
							f.write(f"{flag[w-1]} {current_index[workers_cnt]}\n")
		workers_cnt += 1

	workers_cnt = 0
	for worker in workers_this_round:
		flag[worker-1] = current_index[workers_cnt]
		workers_cnt += 1
						

		
g = Digraph("nodes.txt", "edges.txt")
#g.visualization("test2.png")
g.visualization(f"{log_folder}/path.png")