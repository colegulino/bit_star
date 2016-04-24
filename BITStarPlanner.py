import numpy, random
from RRTTree import RRTTree

class BITStarPlanner(object):

    '''
    Object initializer function
    '''
    def __init__(self, planning_env, visualize):
        self.planning_env = planning_env
        self.visualize = visualize
        self.vertex_queue = [] # self.vertex_queue = node_id
        self.edge_queue = [] # self.edge_queue = (sid, eid)
        self.samples = dict() # self.edge_queue[node_id] = config
        self.g_scores = dict() # self.g_scores[node_id] = g_score
        self.f_scores = dict() # self.f_scores[node_id] = f_score
        self.r = float("inf") # radius
        self.v_old = [] # old vertices 

    '''
    Main Implementation for getting a plan
    '''
    def Plan(self, start_config, goal_config, epsilon = 0.001):
        
        # Initialize plan
        plan = []
        self.start_config = start_config
        self.goal_config = goal_config
        self.start_id = self.planning_env.discrete_env.ConfigurationToNodeId(start_config)
        self.goal_id = self.planning_env.discrete_env.ConfigurationToNodeId(goal_config)
        self.tree = RRTTree(self.planning_env, start_config) # initialize tree

        # Initialize plot
        if self.visualize and hasattr(self.planning_env, 'InitializePlot'):
            self.planning_env.InitializePlot(goal_config)

        # Add the goal to the samples
        self.samples[self.goal_id] = self.goal_config
        self.g_scores[self.goal_id] = float("inf")
        self.f_scores[self.goal_id] = 0

        # Add the start id to the tree
        self.tree.AddVertex(self.start_config)
        self.g_scores[self.start_id] = 0
        self.f_scores[self.start_id] = self.planning_env.ComputeHeuristicCost(self.start_id, self.goal_id)

        # Specifies the number of iterations
        iterations = 0
        max_iter = 100000

        print "Start ID: ", self.start_id
        print "Goal ID: ", self.goal_id

        self.r = 2.0
        # Initalize the g_score of the goal
        # run until done
        found_goal = False
        while(iterations < max_iter):
            # Add the start of a new batch
            if len(self.vertex_queue) == 0 and len(self.edge_queue) == 0:
                if iterations == 0:
                    self.samples.update(self.Sample(m=100, c_max=self.g_scores[self.goal_id]))

                if found_goal:
                    self.r *= 0.8
                    print "New Batch"
                    # Prune the tree
                    self.Prune(self.g_scores[self.goal_id])
                    # Add values to the samples
                    self.samples.update(self.Sample(m=100, c_max=self.g_scores[self.goal_id]))
                    print "Radius: ", self.r

                # Make the old vertices the new vertices
                self.v_old += self.tree.vertices.keys()
                # Add the vertices to the vertex queue
                self.vertex_queue += self.tree.vertices.keys()
                # Change the size of the radius
                #self.r = self.radius(len(self.tree.vertices) + len(self.samples))

            # Expand the best vertices until an edge is better than the vertex
            while(self.BestVertexQueueValue() <= self.BestEdgeQueueValue()):
                self.ExpandVertex(self.BestInVertexQueue())

            print "Vertex Queue: ", self.vertex_queue
            # Add the best edge to the tree 
            best_edge = self.BestInEdgeQueue()
            self.edge_queue.remove(best_edge)
            # See if it can improve the solution
            estimated_cost_of_vertex = self.g_scores[best_edge[0]] + self.planning_env.ComputeDistance(best_edge[0], best_edge[1]) + self.planning_env.ComputeHeuristicCost(best_edge[1], self.goal_id)
            estimated_cost_of_edge = self.planning_env.ComputeDistance(self.start_id,best_edge[0]) + self.planning_env.ComputeDistance(best_edge[0], best_edge[1]) + self.planning_env.ComputeHeuristicCost(best_edge[1], self.goal_id)
            actual_cost_of_edge = self.g_scores[best_edge[0]] + self.planning_env.ComputeDistance(best_edge[0], best_edge[1])

            if(estimated_cost_of_vertex < self.g_scores[self.goal_id]):
                if(estimated_cost_of_edge < self.g_scores[self.goal_id]):
                    if(actual_cost_of_edge < self.g_scores[self.goal_id]):
                        # Connect
                        first_config = self.planning_env.discrete_env.NodeIdToConfiguration(best_edge[0])
                        next_config = self.planning_env.discrete_env.NodeIdToConfiguration(best_edge[1])
                        path = self.con(first_config, next_config)
                        if path == None or len(path) == 0: # no path
                            continue
                        next_config = path[len(path)-1,:]
                        last_config_in_path_id = self.planning_env.discrete_env.ConfigurationToNodeId(next_config)
                        best_edge = (best_edge[0], last_config_in_path_id)
                        if(best_edge[1] in self.tree.vertices.keys()):
                            edges_to_delete = []
                            for sid, eid in self.tree.edges.iteritems():
                                if eid == best_edge[1]:
                                    edges_to_delete.append(sid)
                            for sid in edges_to_delete:
                                del self.tree.edges[sid]
                        else:
                            try:
                                del self.samples[best_edge[1]]
                            except(KeyError):
                                pass
                            eid = self.tree.AddVertex(next_config)
                        if eid == self.goal_id:
                            found_goal = True

                        self.tree.AddEdge(best_edge[0], best_edge[1])

                        g_score = self.planning_env.ComputeDistance(best_edge[0], best_edge[1])
                        self.g_scores[best_edge[1]] = g_score + self.g_scores[best_edge[0]]
                        self.f_scores[best_edge[1]] = g_score + self.planning_env.ComputeHeuristicCost(best_edge[1], self.goal_id)

                        if self.visualize:
                            self.planning_env.PlotEdge(first_config, next_config)

                        for edge in self.edge_queue:
                            if edge[0] == best_edge[1]:
                                if self.g_scores[edge[0]] + self.planning_env.ComputeDistance(edge[0], best_edge[1]) >= self.g_scores[self.goal_id]:
                                    self.edge_queue.remove((edge[0], best_edge[1])) 
                            if(edge[1] == best_edge[1]):
                                if(self.g_scores[edge[1]] + self.planning_env.ComputeDistance(edge[1], best_edge[1]) >= self.g_scores[self.goal_id]):
                                    self.edge_queue.remove((edge[1], best_edge[1])) 
            else:
                self.edge_queue = []
                self.vertex_queue = []

            iterations += 1

    '''
    Function to expand a vertex
    '''
    def ExpandVertex(self, vid):
        # Remove vertex from vertex queue
        #print "Expanding vertex"
        self.vertex_queue.remove(vid)

        # Get the current configure from the vertex 
        curr_config = numpy.array(self.tree.vertices[vid])

        # Get a nearest value in vertex for every one in samples where difference is less than the radius
        possible_neighbors = [] # possible sampled configs that are within radius
        #print "Samples to expand: ", self.samples 
        for sample_id, sample_config in self.samples.iteritems():
            sample_config = numpy.array(sample_config)
            if(numpy.linalg.norm(sample_config - curr_config,2) <= self.r):
                possible_neighbors.append((sample_id, sample_config))

        # Add an edge to the edge queue if the path might improve the solution
        for neighbor in possible_neighbors:
            sample_id = neighbor[0]
            sample_config = neighbor[1]
            estimated_f_score = self.planning_env.ComputeDistance(self.start_id, vid) + self.planning_env.ComputeDistance(vid, sample_id) + self.planning_env.ComputeHeuristicCost(sample_id, self.goal_id)
            if estimated_f_score < self.g_scores[self.goal_id]:
                self.edge_queue.append((vid, sample_id))


        #print "Edge queue after expansion on neighbors: ", self.edge_queue

        # Add the vertex to the edge queue
        if vid not in self.v_old:
            possible_neighbors = []
            for vid, v_config in self.tree.vertices.iteritems():
                v_config = numpy.array(v_config)
                if(numpy.linalg.norm(v_config - curr_config,2) <= self.r):
                    possible_neighbors.append((vid, v_config))

            # Add an edge to the edge queue if the path might improve the solution
            for neighbor in possible_neighbors:
                sample_id = neighbor[0]
                sample_config = neighbor[1]
                estimated_f_score = self.planning_env.ComputeDistance(self.start_id, vid) + self.planning_env.ComputeDistance(vid, sample_id) + self.planning_env.ComputeHeuristicCost(sample_id, self.goal_id)
                if estimated_f_score < self.g_scores[self.goal_id] and (self.g_scores[vid] + self.planning_env.ComputeDistance(vid,sample_id)) < self.g_scores[sample_id]:
                    self.edge_queue.append((vid, sample_id))

        #print "Edge queue after expansion on vertices: ", self.edge_queue

    '''
    Function to prune the tree
    '''
    def Prune(self, c):
        # Remove samples whose estmated cost to goal is > c
        #print "Samples", self.samples
        self.samples = {node_id:config for node_id, config in self.samples.iteritems() if self.planning_env.ComputeDistance(self.start_id, node_id) + self.planning_env.ComputeHeuristicCost(node_id, self.goal_id) <= c}
        # Remove vertices whose estimated cost to goal is > c
        self.tree.vertices = {node_id:config for node_id, config in self.tree.vertices.iteritems() if self.f_scores[node_id] <= c}
        # Remove edge if either vertex connected to its estimated cost to goal is > c
        self.tree.edges = {eid:sid for eid, sid in self.tree.edges.iteritems() if self.f_scores[eid] < c and self.f_scores[sid] <= c}
        # Add vertices to samples if its g_score is infinity
        new_samples = {node_id:config for node_id, config in self.tree.vertices.iteritems() if self.g_scores[node_id] == float("inf")}
        for node_id, config in new_samples:
            if node_id not in self.samples.keys():
                self.samples[node_id] = config
        # Remove vertices whose g_score is infinity
        self.tree.vertices = {node_id:config for node_id, config in self.tree.vertices.iteritems() if self.g_scores[node_id] != float("inf")}

    '''
    Function to extend between two configurations
    '''
    def ext(self, tree, random_config):
        # Get the nearest configuration to this
        sid, nearest_config = tree.GetNearestVertex(random_config)
        # Get the interpolation between the two
        path = self.planning_env.Extend(nearest_config, random_config)
        # Return only the first two parts in the path
        if(path == None):
            return path, sid, nearest_config
        else:
            return path[0:2,:], sid, nearest_config

    '''
    Function to connnect two configurations
    '''
    def con(self, start_config, end_config):
        # Return the whole path to the end
        return self.planning_env.Extend(start_config, end_config)

    '''
    Function to get the new radius of the r-disk
    '''
    def radius(self, q):
        eta = 2.0 # tuning parameter
        dimension = len(self.planning_env.lower_limits) + 0.0 # dimension of problem
        space_measure = self.planning_env.space_measure # volume of the space
        unit_ball_measure = self.planning_env.unit_ball_measure # volume of the dimension of the unit ball

        min_radius = eta * 2.0 * pow((1.0 + 1.0/dimension) * (space_measure/unit_ball_measure), 1.0/dimension) 
        return min_radius * pow(numpy.log(q)/q, 1/dimension) 

    def GetNearestSample(self, config):
        dists = dict()
        for index in self.samples.keys():
            if index==self.planning_env.discrete_env.ConfigurationToNodeId(self.start_config):
                dists[index]=999
                pass
            dists[index] = self.planning_env.ComputeDistance(self.planning_env.discrete_env.ConfigurationToNodeId(config), index)

        # vid, vdist = min(dists.items(), key=operator.itemgetter(0))
        sample_id = min(dists, key=dists.get)

        return sample_id, self.samples[sample_id]

    def Sample(self, m, c_max = float("inf")):
        new_samples = dict()
        if c_max < float("inf"):
            c_min = self.planning_env.ComputeDistance(self.start_config, self.goal_config)
            x_center = (self.start_config + self.goal_config) / 2
            # Get a random sample form the unit ball
            X_ball = self.SampleUnitNBall(m)
            # scale the unit ball
            scale = self.planning_env.GetEllipsoidScale(c_max, c_min)
            points_scale = numpy.dot(X_ball, scale)
            # Translate them to the center
            points_trans = points_scale + x_center 
            # generate the dictionary
            for point in points_trans:
                node_id = self.planning_env.discrete_env.ConfigurationToNodeId(numpy.array(point))
                new_samples[node_id] = numpy.array(point)
        else:
            # Initially just uniformly sample
            for i in xrange(0, m + 1):
                random_config = self.planning_env.GenerateRandomConfiguration()
                random_id = self.planning_env.discrete_env.ConfigurationToNodeId(random_config)
                new_samples[random_id] = random_config
        return new_samples

    def SampleUnitNBall(self, m):
        points = numpy.random.uniform(-1, 1, [m*2, self.planning_env.dimension])
        points = list(points)
        points = [point for point in points if numpy.linalg.norm(point,2) < 1]
        points = numpy.array(points)
        points = list(points)
        points = [point for point in points if self.planning_env.ValidConfig(point)]
        points = numpy.array(points)
        print "Shape of points: ", numpy.shape(points)
        return points[0:m,:]

    def BestVertexQueueValue(self):
        # Return the best value in the Queue by score g_tau[v] + h[v]
        if(len(self.vertex_queue) == 0): # Edge Case
            return float("inf")
        #print "Vertex Queue before: ", self.vertex_queue
        values = [self.g_scores[v] + self.planning_env.ComputeHeuristicCost(v,self.goal_id) for v in self.vertex_queue]
        values.sort()
        #print "Vertex queue after sorting: ", self.vertex_queue
        #print "Vertex queue G_score: ", self.g_scores[self.vertex_queue[0]] + self.planning_env.ComputeHeuristicCost(self.vertex_queue[0], self.goal_id)
        #print "Values in vertex queue ", values
        return values[0]

    def BestEdgeQueueValue(self):
        if(len(self.edge_queue) == 0): # Edge case
            return float("inf")
        # Return the best value in the queue by score g_tau[v] + c(v,x) + h(x)
        values = [self.g_scores[e[0]] + self.planning_env.ComputeDistance(e[0], e[1]) + self.planning_env.ComputeHeuristicCost(e[1], self.goal_id) for e in self.edge_queue]
        values.sort(reverse=True)
        return values[0]

    def BestInEdgeQueue(self):
        # Return the best value in the edge queue
        #print "Edge queue: ", self.edge_queue
        e_and_values = [(e[0], e[1], self.g_scores[e[0]] + self.planning_env.ComputeDistance(e[0], e[1]) + self.planning_env.ComputeHeuristicCost(e[1], self.goal_id)) for e in self.edge_queue]
        #print "Values in the edge queue before sorting: ", e_and_values
        e_and_values = sorted(e_and_values, key=lambda x : x[2])
        return (e_and_values[0][0], e_and_values[0][1])

    def BestInVertexQueue(self):
        # Return the besst value in the vertex queue 
        #print "Best In Vertex Queue before ", self.vertex_queue
        v_and_values = [(v,self.g_scores[v] + self.planning_env.ComputeHeuristicCost(v,self.goal_id)) for v in self.vertex_queue]
        v_and_values = sorted(v_and_values, key=lambda x : x[1])
        #print "Values after: ", v_and_values

        return v_and_values[0][0]

