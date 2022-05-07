import csv
from os import listdir

time = '04162022_195253'

with open('test.csv', mode='w') as file:
	writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
	writer.writerow(['comm', 'device', 'is_worker', 'ASR'])

	log_folder = f"logs/{time}"
	all_rounds_log_files = sorted([f for f in listdir(log_folder) if f.startswith('comm')], key=lambda x: int(x.split('.')[0].split('_')[-1]))
	rounds = len(all_rounds_log_files) - 1

	for i in range(1, rounds+1):
		for id in range(1, 11):
			with open(f"{log_folder}/comm_{i}/device_{id}_asr_comm_{i}.txt", 'r') as file:
				lines_list = file.read().split("\n")
				if lines_list[1].split(': ')[-1] == 'worker':
					writer.writerow([i, id, 1, float(lines_list[2].split(': ')[-1])])
				else:
					writer.writerow([i, id, 0, float(lines_list[2].split(': ')[-1])])