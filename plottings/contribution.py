from matplotlib import pyplot as plt

data_1 = [0.1667, 0.1667, 0.1667, 0.1667, 0.1667, 0.1667]
data_2 = [0.357, 0.0714, 0.0714, 0.0714, 0.0714, 0.357]

x = ['d_1', "d_2", "d_3", 'd_4', 'd_5', 'd_6']

fig = plt.figure(figsize=(4,3), dpi=300)
plt.bar(range(1,7), data_2)
plt.xlabel("Device id")
plt.ylabel("Contribution ratio")
plt.ylim(0,1.0)
plt.tight_layout()
#plt.xticks(x)
plt.savefig("test_2.png", dpi=300)
plt.show()