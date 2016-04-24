class RRTConnectPlanner(object):

    '''
    Object initializer function
    '''
    def __init__(self, planning_env, visualize):
        self.planning_env = planning_env
        self.visualize = visualize
        self.vertex_queue = [] # self.vertex_queue = node_id
        self.edge_queue = [] # self.edge_queue = (sid, eid)
        self.samples = dict() # self.edge_queue[node_id] = config
        self.tree = RRTTree(self.planning_env, start_config) # initialize tree
        self.g_scores = dict() # self.g_scores[node_id] = g_score
        self.f_scores = dict() # self.f_scores[node_id] = f_score
        self.r = float("inf") # radius
        self.v_old = dict() # old vertices 

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

        # Initialize plot
        if self.visualize and hasattr(self.planning_env, 'InitializePlot'):
            self.planning_env.InitializePlot(goal_config)

        # Add the goal to the samples
        self.samples[self.goal_id] = self.goal_config

        # Specifies the number of iterations
        iterations = 1
        max_iter = 100

        # Initalize the g_score of the goal
        self.g_scores[self.goal_id] = float("inf")
        # run until done
        while(iterations < max_iter):
            if len(vertex_queue) == 0 and len(edge_queue) == 0:
                # Prune the tree
                self.Prune(self.g_scores[self.goal_id])
                # Add values to the samples
                self.Sample(m=10)
                # Make the old vertices the new vertices
                self.v_old = self.tree.vertices
                # Change the size of the radius
                self.r = radius(len(self.tree.vertices) + len(self.samples))

            # Expand the best vertices until an edge is better than the vertex



    '''
    Function to expand a vertex
    '''
    def ExpandVertex(self, vid):
        try:
            # Remove vertex from vertex queue
            self.vertex_queue.remove(vid)

            # Get the current configure from the vertex 
            curr_config = numpy.array(self.tree.vertices[vid])

            # Get a nearest value in vertex for every one in samples where difference is less than the radius
            possible_neighbors = [] # possible sampled configs that are within radius
            for sample_id, sample_config in self.samples.iteritems():
                sample_config = numpy.array(sample_config)
                if(numpy.linalg.norm(sample_config - curr_config,2) <= self.r):
                    possible_neighbors.append((sample_id, sample_config))

            # Add an edge to the edge queue if the path might improve the solution
            for neighbor in possible_neighbors:
                sample_id = neighbor[0]
                sample_config = neighbor[1]
                estimated_f_score = self.planning_env.ComputeDistance(self.start_id, vid) + \
                 self.planning_env.ComputeDistance(vid, sample_id) + \ 
                 self.planning_env.ComputeHeuristicCost(sample_id, self.goal_id)
                if estimated_f_score < g_score[self.goal_id]:
                    self.edge_queue.append(vid, sample_id)

            # Add the vertex to the edge queue
            if vid not in v_old.keys():
                possible_neighbors = []
                for vid, v_config in self.tree.vertices.iteritems():
                    v_config = numpy.array(v_config)
                    if(numpy.linalg.norm(v_config - curr_config,2) <= self.r):
                        possible_neighbors.append((vid, v_config))

                # Add an edge to the edge queue if the path might improve the solution
                for neighbor in possible_neighbors:
                    sample_id = neighbor[0]
                    sample_config = neighbor[1]
                    estimated_f_score = self.planning_env.ComputeDistance(self.start_id, vid) + \
                     self.planning_env.ComputeDistance(vid, sample_id) + \ 
                     self.planning_env.ComputeHeuristicCost(sample_id, self.goal_id)
                    if estimated_f_score < g_score[self.goal_id] and (self.g_scores[vid] + self.planning_env.ComputeDistance(vid,sample_id)) < g_scores[sample_id]:
                        self.edge_queue.append(vid, sample_id)

        except(KeyError):
            print "Couldn't find key"
            return

    '''
    Function to prune the tree
    '''
    def Prune(self, c):
        try:
            # Remove samples whose estmated cost to goal is > c
            self.samples = {node_id:f_score for node_id, f_score in self.samples.iteritems() if f_score < c}
            # Remove vertices whose estimated cost to goal is > c
            self.tree.vertices = {node_id:config for node_id, config in self.tree.vertices.iteritems() if self.f_scores[node_id] < c}
            # Remove edge if either vertex connected to its estimated cost to goal is > c
            self.tree.edges = {eid:sid for eid, sid in self.tree.edges.iteritems() if self.f_scores[eid] < c and self.f_scores[sid] < c}
            # Add vertices to samples if its g_score is infinity
            new_samples = {node_id:config for node_id, config in self.tree.vertices.iteritems() if self.g_scores[node_id] = float("inf")}
            for node_id, config in new_samples:
                if node_id not in self.samples.keys():
                    self.samples[node_id] = config
            # Remove vertices whose g_score is infinity
            self.tree.vertices = {node_id:config for node_id, config in self.tree.vertices.iteritems() if self.g_scores[node_id] != float("inf")}
        except(KeyError):
            print "Couldn't find key"
            return

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
    def con(self, tree, random_config):
        # Get the nearest configuration to this
        sid, nearest_config = tree.GetNearestVertex(random_config)
        # Return the whole path to the end
        return self.planning_env.Extend(nearest_config, random_config), sid, nearest_config

    '''
    Function to get the new radius of the r-disk
    '''
    def radius(self, q):
        eta = 2.0 # tuning parameter
        dimension = len(self.planning_env.lower_limits) # dimension of problem
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
            points_scale = np.dot(points_d, scale)
            # Translate them to the center
            points_trans = points_scale + x_center 
            # generate the dictionary
            for point in points_trans:
                node_id = self.planning_env.discrete_env.ConfigurationToNodeId(points_trans)
                new_samples[node_id] = numpy.array(point)
        else:
            # Initially just uniformly sample
            for i in xrange(0, m + 1):
                random_config = self.planning_env.GenerateRandomConfiguration()
                random_id = self.planning_env.discrete_env.ConfigurationToNodeId(random_config)
                new_samples[random_id] = random_config
        return new_samples

    def SampleUnitNBall(self, m):
       points = np.random.uniform(-1, 1, [m*2, self.planning_env.dimension])
       points = list(points)
       points = [point for point in points if np.linalg.norm(point,2) < 1]
       points = numpy.array(points)
       points = list(points)
       points = [point for point in points if self.planning_env.ValidConfig(point)]
       return points[0:m, self.planning_env.dimension]

 


