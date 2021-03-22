## Graph Cut Based Active Contours

	function ***segmentation***(*image*, *c0*):
		**input**: original image (as array) *image* and initial contour *c0*
		**output**: final segmentation contour
		
		*G* = ***image_to_graph***(*image*)
		*i* <- 0
		*cp* <- *c0*
		**repeat**
			*ic*, *oc* = ***dilation***(*cp*)
			*Gi* = ***vertex_identification***(*ic*, *oc*, *G*)
			*ci* = ***min_cut***(*Gi*)
			**if** *norm*(*ci*-*cp*) < 1e-5 **then**
				**return** *ci*
			**else**
				*i* = *i* + 1
				*cp* = *ci*

	function ***image_to_graph***(*image*):
		**input**: original image (as array) *image*
		**output**: weighted adjacency matrix *A* of graph representation of *image*
		
		*m*, *n* <- size of *image*
		*A* <- array of zeros of size (*m* x *n*) by (*m* x *n*)
		**for** *i* = 0, 1, 2, ..., m **do**
			**for** *j* = 0, 1, 2, ..., n **do**
				**for each** neighbor of pixel (*i*,*j*) **do**
					1. calculate capacity of edge between pixel (*i*,*j*) and neighbor
					2. set entry in *A* corresponding to pixel (*i*,*j*) and neighbor equal to calculated capacity
		**return** *A*

	function ***dilation***(*c*):
		**input**: contour *c*
		**output**: inner contour *ic* and outer contour *oc* of dilated contour neighborhood
		
		*dilated* <- contour *c* expanded into neighborhood (use function or morphological convolution)
		*ic* <- inside contour of *dilated*
		*oc* <- outside contour of *dilated*
		**return** *ic*, *oc*

	function ***vertex_identification***(*ic*, *oc*, *A*):
		**input**: inner contour *ic*, outer contour *oc*, adjacency matrix *A* for original graph
		**output**: graph of image where vertices in input contours identified as single vertex
		
		*m*, *n* <- size of *image*
		*edges_inner* = array of zeros of size (*m* x *n*) by 1
		*edges_outer* = array of zeros of size (*m* x *n*) by 1
		**for** vertex *u* in *A* **do**
			**for** vertex *v* in *ic* **do**
				**if** exists edge (*u*,*v*) in *A* **then**
					*edges_inner*[index corresponding to *u*] += capacity of edge (*u*,*v*)
			**for** vertex *v* in *oc* **do**
				**if** exists edge (*u*,*v*) in *A* **then**
					*edges_outer*[index corresponding to *u*] += capacity of edge (*u*,*v*)
		*Ai* <- copy of *A*
		augment *Ai* with *edges_inner* and *edges_outer* as new rows
		augment *Ai* with transposes of *edges_inner* and *edges_outer* as new columns
		delete from *Ai* rows and columns corresponding to vertices in *ic*
		delete from *Ai* rows and columns corresponding to vertices in *oc*
		**return** *Ai*

	function ***min_cut***(*G*):
		**input**: graph *G* representing an image
		**output**: set of vertices on the "sink" side of a minimum cut in *G*
		
		find partition of vertices in *G* between source (*ic* vertex) and sink (*oc* vertex) (use minimum cut function where cut value determined by capacity)
		**return** set of vertices on sink side of the cut