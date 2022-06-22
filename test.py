from DatasetLoad import DatasetLoad
import matplotlib.pyplot as plt
import torch
import numpy as np
import helper

# fig = plt.figure(figsize=(4,3), dpi=300)
# img = np.zeros((784,))
# poison_img = helper.add_pixel_pattern(img, -1)
# poison_img = poison_img.reshape(28,28)

# plt.xticks([])
# plt.yticks([])
# plt.xlabel("L1 norm = 16", fontsize=20)
# plt.imshow(poison_img, cmap='binary')
# plt.savefig("test.png")



# for i in range(10):
#     image = torch.tensor(train_data[i])
#     new_image = image.reshape(28,28)
#     plt.xticks([])
#     plt.yticks([])
#     plt.axis('off')
#     plt.imshow(new_image, cmap='binary')
#     plt.savefig(f"data_{i}.png")
#     plt.show()



# import networkx as nx
# import matplotlib.pyplot as plt

# g = nx.DiGraph()

# g.add_node(1)
# g.add_nodes_from([2,3,4])
# g.nodes()

# nx.draw(g)

# plt.savefig("test.png")
# plt.show()

