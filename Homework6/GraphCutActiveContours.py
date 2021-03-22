import numpy as np
import cv2
import matplotlib.pyplot as plt
import networkx as nx

class GraphCutActiveContours:
	def __init__(self, image):
		self.image = image
		self.m, self.n = image.shape

	'''
	perform graph cut based active contours segmentation
	'''
	def segmentation(self, c0, max_iter = None, visualize=False, show_graphs=False):
		# make graph for image
		self.image_to_graph()
		if(visualize and show_graphs):
			plt.imshow(self.A)
			plt.title("adjacency matrix for graph G")
			plt.show()
			# subset attribute allows for better visualization of graph as grid
			for n in list(self.G.nodes):
				self.G.nodes[n]['subset'] = n % self.m # row number
			nx.draw(self.G, with_labels=True, pos=nx.multipartite_layout(self.G))
			plt.show()
		i = 0
		contour_list = [c0]
		cp = c0
		ci = c0
		while True:
			if(visualize):
				print("ITERATION %d" % i)
			# 1. Dilate current contour ci into its contour neighborhood CN(ci) with an inner contour ICi and an outer contour OCi.
			dilated, contours = self.dilation(cp)
			if(len(contours) == 1):
				ci_image = np.zeros(cp.shape, dtype=np.uint8)
				xs = [x[0][0] for x in contours[0]]
				ys = [x[0][1] for x in contours[0]]
				for j in range(len(xs)):
					ci_image[xs[j], ys[j]] = 1
				if(visualize):
					plt.imshow(ci_image, cmap='gray')
					plt.show()
				return ci_image
			if(visualize):
				plt.subplot(1,2,1)
				plt.imshow(dilated, cmap = 'gray')
				plt.title("dilated contour")
				xs = [x[0][0] for x in contours[0]]
				ys = [x[0][1] for x in contours[0]]
				plt.subplot(1,2,2)
				plt.imshow(dilated, cmap = 'gray')
				plt.plot(xs, ys)
				xs = [x[0][0] for x in contours[1]]
				ys = [x[0][1] for x in contours[1]]
				plt.plot(xs, ys)
				plt.title("inner and outer contour")
				plt.show()
			# 2. Identify all the vertices corresponding to the inner contour as a single source si and identify all the vertices corresponding to the outer contour as a single sink ti to obtain a new graph Gi.
			Gi, Ai = self.vertex_identification(contours[0], contours[1])
			if(visualize and show_graphs):
				plt.imshow(Ai)
				plt.title("adjacency matrix for new graph Gi")
				plt.show()
				# subset attribute allows for better visualization of graph as grid
				for n in list(Gi.nodes):
					if n != 's' and n != 't':
						Gi.nodes[n]['subset'] = n % self.m + 1 # row number
				Gi.nodes['s']['subset'] = int(0)
				Gi.nodes['t']['subset'] = int(self.n+1)
				pos=nx.multipartite_layout(Gi)
				nx.draw(Gi, with_labels=True, pos=pos)
				edge_labels = nx.get_edge_attributes(Gi,'capacity') # key is edge, pls check for your case
				nx.draw_networkx_edge_labels(Gi,pos,edge_labels=edge_labels,font_color='red')
				plt.show()
			# 3. Compute the s–t minimum cut MC(Gi,si,ti) to obtain a new contour ci+1
			ci = self.min_cut(Gi)
			ci_image = np.zeros(cp.shape, dtype=np.uint8)
			# xs = [x[0][0] for x in ci]
			# ys = [x[0][1] for x in ci]
			xs = [x[0] for x in ci]
			ys = [x[1] for x in ci]
			for j in range(len(xs)):
				ci_image[xs[j], ys[j]] = 1
			if(visualize):
				# plt.subplot(1,3,1)
				# plt.imshow(self.image, cmap = 'gray')
				# plt.plot(xs, ys)
				plt.title("new contour")
				plt.subplot(1,2,1)
				plt.imshow(ci_image, cmap='gray')
				plt.title("new contour")
				plt.subplot(1,2,2)
				plt.imshow(cp, cmap='gray')
				plt.title("previous contour")
				plt.show()
			# 4. Terminate the algorithm if a resulting contour reoccurs, otherwise set i=i+1 and return to step 1.
			if np.linalg.norm(cp-ci_image) < 1e-5:
				break
			else:
				contour_list.append(ci_image)
				cp = ci_image
				if max_iter is not None and i >= max_iter:
					break
				i = i + 1
		
		return ci_image


	'''
	perform dilation operation on contour to expand into contour neighborhood
	'''
	def dilation(self, contour):
		kernel = np.ones((3,3))
		dilated = cv2.dilate(contour.astype(np.uint8), kernel, iterations = 1)
		dilated = dilated.astype(np.uint8)
		# get inner and outer contours of dilated contour neighborhood
		contours, hierarchy = cv2.findContours(dilated.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
		return dilated, contours

	'''
	identify vertices in inner and outer contours into single vertices
	return new graph Gi and indices for inner and outer vertices
	'''
	def vertex_identification(self, inner, outer):
		idx_inner = []
		idx_outer = []
		for x in inner:
			idx_inner.append(x[0][0]*self.m+x[0][1])
		for x in outer:
			idx_outer.append(x[0][0]*self.m+x[0][1])

		new_size = self.m*self.n-len(idx_inner)-len(idx_outer)+2
		
		# add edges from new vertex to other vertices connected to vertex in sets
		edges_inner = np.zeros((self.m*self.n))
		edges_outer = np.zeros((self.m*self.n))
		capacity_inner = np.zeros((self.m*self.n))
		capacity_outer = np.zeros((self.m*self.n))
		inner_capacity = 0
		outer_capacity = 0
		for i in range(self.m*self.n):
			for idx in idx_inner:
				if self.A[idx,i] == 1:
					edges_inner[i] = 1
					capacity_inner[i] += self.G[idx][i]['capacity']
			for idx in idx_outer:
				if self.A[idx,i] == 1:
					edges_outer[i] = 1
					capacity_outer[i] += self.G[idx][i]['capacity']
					
		Ai = self.A.copy()
		
		# add new vertices
		Ai = np.hstack((Ai, edges_inner.reshape(-1,1), edges_outer.reshape(-1,1)))
		edges_inner_hz = np.hstack((np.transpose(edges_inner),np.asarray([0,0])))
		edges_outer_hz = np.hstack((np.transpose(edges_outer),np.asarray([0,0])))
		Ai = np.vstack((Ai, edges_inner_hz, edges_outer_hz))

		# remove identified vertices
		Ai = np.delete(Ai, idx_inner, 0)
		Ai = np.delete(Ai, idx_outer, 0)
		Ai = np.delete(Ai, idx_inner, 1)
		Ai = np.delete(Ai, idx_outer, 1)

		# work on graph Gi and calculate appropriate accumulated capacity
		Gi = nx.Graph()
		Gi.add_edges_from(self.G.edges) # all old edges and nodes
		# add old capacities
		for u,v in Gi.edges:
			Gi[u][v]['capacity'] = self.G[u][v]['capacity']
		# add two new nodes
		Gi.add_node("s")
		Gi.add_node("t")
		# add new edges with capacities
		for i in range(self.m*self.n):
			if edges_inner[i] == 1:
				Gi.add_edge("s", i, capacity=capacity_inner[i])
			if edges_outer[i] == 1:
				Gi.add_edge("t", i, capacity=capacity_outer[i])
		# remove old vertices (and edges)
		Gi.remove_nodes_from(idx_inner)
		Gi.remove_nodes_from(idx_outer)
		
		return Gi, Ai


	'''
	find minimum cut in graph G between source and sink (identified vertices)
	return new contour based on sink side of cut
	'''
	def min_cut(self, G):
		cut_value, partition = nx.minimum_cut(G, "s", "t")
		# find new contour based on min cut
		cut = np.zeros(self.image.shape)
		all_points = []
		for i in partition[1]:
			if i != 't':
				x = int(i/self.m)
				y = i%self.m
				cut[x,y] = 1
				all_points.append((x,y))
		cut = cut.astype(np.uint8)
		new_contour = []
		for j in [x[1] for x in all_points]:
			this_col = [x[0] for x in all_points if x[1]==j]
			new_contour.append((max(this_col),j))
			new_contour.append((min(this_col),j))
		for i in [x[0] for x in all_points]:
			this_row = [x[1] for x in all_points if x[0]==i]
			new_contour.append((i,max(this_row)))
			new_contour.append((i,min(this_row)))
		# contours, hierarchy = cv2.findContours(cut.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
		# new_contour = contours[0]
		return new_contour
		# return cut

	'''
	build graph (adjacency matrix) from image
	vertices are pixels in the image
	there is an edge between a pixel and its 8 neighbors (above, below, right, left, and 4 diagonals)

	adjacency matrix structure:
	if image is:
	1 2
	3 4
	adjacency matrix is:
	 1 2 3 4 
	1
	2
	3
	4
	'''
	def image_to_graph(self):
		A = np.zeros((self.m*self.n, self.m*self.n))
		
		# handle corners
		# top left: only right, below, and SE diagonal; idx = 0
		idx = 0
		A[idx,idx+self.m] = 1 # below
		A[idx,idx+1] = 1 # right
		A[idx,idx+self.m+1] = 1 # SE diagonal
		# symmetry
		A[idx+self.m,idx] = 1 # below
		A[idx+1,idx] = 1 # right
		A[idx+self.m+1,idx] = 1 # SE diagonal
		
		# top right: only left, below, and SW diagonal; idx = n-1
		idx = self.n-1
		A[idx,idx+self.m] = 1 # below
		A[idx,idx-1] = 1 # left
		A[idx,idx+self.m-1] = 1 # SW diagonal
		# symmetry
		A[idx+self.m,idx] = 1 # below
		A[idx-1,idx] = 1 # left
		A[idx+self.m-1,idx] = 1 # SW diagonal
		
		# bottom left: only right, above, and NE diagonal; idx = (m-1)*m
		idx = (self.m-1)*self.m
		A[idx,idx-self.m] = 1 # above
		A[idx,idx+1] = 1 # right
		A[idx,idx-self.m+1] = 1 # NE diagonal
		# symmetry
		A[idx-self.m,idx] = 1 # above
		A[idx+1,idx] = 1 # right
		A[idx-self.m+1,idx] = 1 # NE diagonal

		# bottom right: only left, above, and NW diagonal; idx = (m-1)*m+(m-1)
		idx = (self.m-1)*self.m+(self.m-1)
		A[idx,idx-self.m] = 1 # above
		A[idx,idx-1] = 1 # left
		A[idx,idx-self.m-1] = 1 # NW diagonal
		# symmetry
		A[idx-self.m,idx] = 1 # above
		A[idx-1,idx] = 1 # left
		A[idx-self.m-1,idx] = 1 # NW diagonal
		
		# handle top row (no above, NW diagonal, or NE diagonal)
		for j in range(1,self.n-1):
			idx = j
			A[idx,idx+self.m] = 1 # below
			A[idx,idx+1] = 1 # right
			A[idx,idx-1] = 1 # left
			A[idx,idx+self.m-1] = 1 # SW diagonal
			A[idx,idx+self.m+1] = 1 # SE diagonal
			# symmetry
			A[idx+self.m,idx] = 1 # below
			A[idx+1,idx] = 1 # right
			A[idx-1,idx] = 1 # left
			A[idx+self.m-1,idx] = 1 # SW diagonal
			A[idx+self.m+1,idx] = 1 # SE diagonal
		
		# handle bottom row (no below, SW diagonal, or SE diagonal)
		for j in range(1,self.n-1):
			idx = (self.m-1)*self.m+j
			A[idx,idx-self.m] = 1 # above
			A[idx,idx+1] = 1 # right
			A[idx,idx-1] = 1 # left
			A[idx,idx-self.m-1] = 1 # NW diagonal
			A[idx,idx-self.m+1] = 1 # NE diagonal
			# symmetry
			A[idx-self.m,idx] = 1 # above
			A[idx+1,idx] = 1 # right
			A[idx-1,idx] = 1 # left
			A[idx-self.m-1,idx] = 1 # NW diagonal
			A[idx-self.m+1,idx] = 1 # NE diagonal
			
		# handle left column (no left, NW diagonal, or SW diagonal)
		for i in range(1,self.m-1):
			idx = i*self.m
			A[idx,idx-self.m] = 1 # above
			A[idx,idx+self.m] = 1 # below
			A[idx,idx+1] = 1 # right
			A[idx,idx-self.m+1] = 1 # NE diagonal
			A[idx,idx+self.m+1] = 1 # SE diagonal
			# symmetry
			A[idx-self.m,idx] = 1 # above
			A[idx+self.m,idx] = 1 # below
			A[idx+1,idx] = 1 # right
			A[idx-self.m+1,idx] = 1 # NE diagonal
			A[idx+self.m+1,idx] = 1 # SE diagonal
			
		# handle right column (no right, NE diagonal, or SE diagonal)
		for i in range(1,self.m-1):
			idx = i*self.m+(self.n-1)
			A[idx,idx-self.m] = 1 # above
			A[idx,idx+self.m] = 1 # below
			A[idx,idx-1] = 1 # left
			A[idx,idx-self.m-1] = 1 # NW diagonal
			A[idx,idx+self.m-1] = 1 # SW diagonal
			# symmetry
			A[idx-self.m,idx] = 1 # above
			A[idx+self.m,idx] = 1 # below
			A[idx-1,idx] = 1 # left
			A[idx-self.m-1,idx] = 1 # NW diagonal
			A[idx+self.m-1,idx] = 1 # SW diagonal
		
		# fill in the rest
		for i in range(1,self.m-1):
			for j in range(1,self.n-1):
				idx = i*self.m+j
				A[idx,idx-self.m] = 1 # above
				A[idx,idx+self.m] = 1 # below
				A[idx,idx+1] = 1 # right
				A[idx,idx-1] = 1 # left
				A[idx,idx-self.m-1] = 1 # NW diagonal
				A[idx,idx-self.m+1] = 1 # NE diagonal
				A[idx,idx+self.m-1] = 1 # SW diagonal
				A[idx,idx+self.m+1] = 1 # SE diagonal
				
				# symmetry
				A[idx-self.m,idx] = 1 # above
				A[idx+self.m,idx] = 1 # below
				A[idx+1,idx] = 1 # right
				A[idx-1,idx] = 1 # left
				A[idx-self.m-1,idx] = 1 # NW diagonal
				A[idx-self.m+1,idx] = 1 # NE diagonal
				A[idx+self.m-1,idx] = 1 # SW diagonal
				A[idx+self.m+1,idx] = 1 # SE diagonal
				
		self.A = A

		self.G = nx.from_numpy_matrix(A)
		for u,v in self.G.edges:
			self.G[u][v]['capacity'] = self.c(u,v)

		return self.A, self.G

	'''
	define weight function between two vertices i and j
	c(i,j) = (g(i,j)+g(j,i))^6
	where g(i,j) = exp(-grad_{ij}(i)/max_{k} grad_{ij}(k))
	'''
	def c(self, i, j):
		return (self.g(i, j)+self.g(j, i))**6

	'''
	g(i,j) = exp(-grad_{ij}(i)/max_{k} grad_{ij}(k))
	where grad_{ij}(k) is the magnitude of image pixel intensity gradient at location k in the direction of i to j
	'''
	def g(self, i, j):
		# determine direction
		if i+1 == j: # right
			kernel=np.asarray([[0,0,0],[0,-1,1],[0,0,0]])
		elif i-1 == j: # left
			kernel=np.asarray([[0,0,0],[1,-1,0],[0,0,0]])
		elif i-self.m == j: # up
			kernel=np.asarray([[0,1,0],[0,-1,0],[0,0,0]])
		elif i+self.m == j: #down
			kernel=np.asarray([[0,0,0],[0,-1,0],[0,1,0]])
		elif i-self.m+1 == j: # up right diagonal
			kernel=np.asarray([[0,0,1],[0,-1,0],[0,0,0]])
		elif i-self.m-1 == j: # up left diagonal
			kernel=np.asarray([[1,0,0],[0,-1,0],[0,0,0]])
		elif i+self.m+1 == j: # down right diagonal
			kernel=np.asarray([[0,0,0],[0,-1,0],[0,0,1]])
		elif i+self.m-1 == j: # down left diagonal
			kernel=np.asarray([[0,0,0],[0,-1,0],[1,0,0]])
		else:
			print("given vertices are not neighbors")
			return None
		
		# calculate
		grad = cv2.filter2D(self.image,-1,kernel)
		x = int(i/self.m)
		y = i%self.m
		return np.exp(-1*grad[x,y]/np.max(grad))