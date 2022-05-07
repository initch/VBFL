import networkx as nx
from matplotlib import pyplot as plt

class Digraph(nx.DiGraph):
	def __init__(self, nodes_log, edges_log):
		super().__init__()
		self.pos = {}
		self.edge_color = []
		self.asr= []
		self.read_log(nodes_log, edges_log)
		
	
	def read_log(self, nodes_log, edges_log):
		with open(nodes_log, 'r') as f:
			lines_list = f.read().split("\n")
			for line in lines_list:
				if line != "":
					# 10 1 M 8 0.4
					index, name, is_malicious, t, asr = line.split(" ")
					self.add_state_node(index, name, int(t), is_malicious, asr)
					self.pos[index] = (int(t), int(name))
					self.asr.append(float(asr))

		with open(edges_log, 'r') as f:
			lines_list = f.read().split("\n")
			for line in lines_list:
				if line != '':
					index_1, index_2 = line.split(" ")
					self.add_event_edge(index_1, index_2)
	
	def add_state_node(self, index, name, t, is_malicious, asr):
		self.add_node(index, name=name, t=t, is_malicious=is_malicious, asr=asr)
	
	def add_event_edge(self, index_1, index_2):
		colors = ['lightsalmon', 'cornflowerblue']
		if nx.get_node_attributes(self, 'is_malicious')[index_1] == 'M':
			infc_type = 0
		else:
			infc_type = 1
		self.add_edge(index_1, index_2, color=colors[infc_type])

	def visualization(self, save_path):
		colors = [self[u][v]['color'] for u,v in self.edges()]
		plt.figure(figsize=(25, 8))
		cmap = plt.cm.get_cmap('Blues')
		node_labels = nx.get_node_attributes(self, 'name')
		nx.draw_networkx_nodes(self, self.pos, node_size=300, cmap=cmap, node_color=self.asr)
		nx.draw_networkx_labels(self, self.pos, labels=node_labels)
		nx.draw_networkx_edges(self, self.pos, edge_color=colors, width=0.8)
		plt.savefig(save_path, dpi=300)
		plt.show()


if __name__ == "__main__":

	g = Digraph("nodes.txt", "edges.txt")
	g.visualization("test.png")