import numpy, time, random, collections
import pylab as pl
from DiscreteEnvironment import DiscreteEnvironment

compare = lambda x, y: collections.Counter(x) == collections.Counter(y)

class SimpleEnvironment(object):
    
    def __init__(self, herb, resolution):
        self.robot = herb.robot
        self.lower_limits = [-5., -5.]
        self.upper_limits = [5., 5.]
        self.discrete_env = DiscreteEnvironment(resolution, self.lower_limits, self.upper_limits)

        # add an obstacle
        self.table = self.robot.GetEnv().ReadKinBodyXMLFile('models/objects/table.kinbody.xml')
        self.robot.GetEnv().Add(self.table)

        table_pose = numpy.array([[ 0, 0, -1, 1.5], 
                                  [-1, 0,  0, 0], 
                                  [ 0, 1,  0, 0], 
                                  [ 0, 0,  0, 1]])
        self.table.SetTransform(table_pose)

        self.p = 0.4

        self.space_measure = 1 # for real numbers
        self.unit_ball_measure = numpy.pi # for 2D space with a radius of 1
        self.C = numpy.array([[1, 0], [0,1]])

    def GetSuccessors(self, node_id):

        successors = []

        # TODO: Here you will implement a function that looks
        #  up the configuration associated with the particular node_id
        #  and return a list of node_ids that represent the neighboring
        #  nodes

        # Implement using an eight connected world
        
        # Function to determine if the configuration is in bounds
        def in_bounds(config):
            return ((config[0] >= self.lower_limits[0] and config[0] <= self.upper_limits[0])) and (config[1] >= self.lower_limits[1] and config[1] <= self.upper_limits[1])

        # Function to determine if the robot is in collision
        def collision_check(x, y):
            transform_t = numpy.eye(4)
            transform_t[0][3] = x
            transform_t[1][3] = y
            self.robot.SetTransform(transform_t)
            collision = self.robot.GetEnv().CheckCollision(self.robot, self.table)
            self.robot.SetTransform(numpy.eye(4))
            return collision

        def extend_collision_check(start_config, end_config):
            #
            # TODO: Implement a function which attempts to extend from 
            #   a start configuration to a goal configuration
            #
            steps = self.ComputeDistance(start_config, end_config)*25 # Get 25 times the number of unit values in the distance space
            x = numpy.linspace(start_config[0], end_config[0], num=steps) # Get the interpolated x values
            y = numpy.linspace(start_config[1], end_config[1], num=steps) # Get the interpolated y values
            # Check the interpolated path for collisions
            for i in xrange(0, len(x)):
                if(collision_check(x[i], y[i]) == True): # If in collision
                    return True
            # If no part of the path is in collision
            return False

        # Function to determine if the perterbed config is within the grid
        def valid_id(id, start_config):
            end_config = self.discrete_env.NodeIdToConfiguration(id)
            #if(extend_collision_check(start_config, end_config)):
            if(collision_check(end_config[0], end_config[1])):
                return False
            return True # if it doesn't go out of bounds

        config = self.discrete_env.NodeIdToConfiguration(node_id)
        direction = [-1, 0, 1]
        perterbed_config = [0] * 2
        for x_dir in direction:
            for y_dir in direction:
                # Get perturbed configuration
                perterbed_config[0] = config[0] + x_dir * self.discrete_env.resolution
                perterbed_config[1] = config[1] + y_dir * self.discrete_env.resolution

                # Check to see if the perturbed configuration is in bounds
                if(in_bounds(perterbed_config)):
                    # Get perturbed node id
                    perterbed_id = self.discrete_env.ConfigurationToNodeId(perterbed_config)
                    if(perterbed_id != node_id):
                        if(valid_id(perterbed_id, config)):
                            # Return the robot to old config
                            transform_t = numpy.eye(4)
                            transform_t[0][3] = config[0]
                            transform_t[1][3] = config[1]
                            self.robot.SetTransform(transform_t)
                            # Append the successor
                            successors.append(perterbed_id)
        return successors

    def ComputeDistance(self, start_id, end_id):

        # TODO: Here you will implement a function that 
        # computes the distance between the configurations given
        # by the two node ids

        # Using Euclidean Distance
        start_config = numpy.array(self.discrete_env.NodeIdToConfiguration(start_id))
        end_config = numpy.array(self.discrete_env.NodeIdToConfiguration(end_id))

        return numpy.linalg.norm(start_config - end_config, 2)

    def ComputeHeuristicCost(self, start_id, goal_id):
        
        # TODO: Here you will implement a function that 
        # computes the heuristic cost between the configurations
        # given by the two node ids

        # Use manhattan distance for grid world 
        start_config = numpy.array(self.discrete_env.NodeIdToConfiguration(start_id))
        goal_config = numpy.array(self.discrete_env.NodeIdToConfiguration(goal_id))

        return numpy.linalg.norm(start_config - goal_config, 1)
        
    def GenerateRandomConfiguration(self):
        config = [0] * 2;
        lower_limits = self.lower_limits
        upper_limits = self.upper_limits
        #
        # TODO: Generate and return a random configuration
        #
        # lower_limits = [-5, -5]
        # upper_limits = [5, 5]
        # config = [0, 0]
        while(True):
            rand_x = numpy.random.uniform(lower_limits[0], upper_limits[0])
            rand_y = numpy.random.uniform(lower_limits[1], upper_limits[1])
            robo = self.robot # Copy the robot
            new_pose = numpy.array([[1, 0, 0, rand_x],
                                    [0, 1, 0, rand_y],
                                    [0, 0, 1, 0     ],
                                    [0, 0, 0, 1     ]])
            robo.SetTransform(new_pose)
            if(self.robot.GetEnv().CheckCollision(robo, self.table) == False):
                config = numpy.array([rand_x, rand_y])
                return numpy.array(config)

    def Extend(self, start_config, end_config,):
        #
        # TODO: Implement a function which attempts to extend from
        #   a start configuration to a goal configuration
        #
        steps = self.ComputeDistance(self.discrete_env.ConfigurationToNodeId(start_config), self.discrete_env.ConfigurationToNodeId(end_config))*25 # Get 25 times the number of unit values in the distance space
        x = numpy.linspace(start_config[0], end_config[0], num=steps) # Get the interpolated x values
        y = numpy.linspace(start_config[1], end_config[1], num=steps) # Get the interpolated y values
        # Check the interpolated path for collisions
        for i in xrange(0, len(x)):
            if(self.collision_check(x[i], y[i]) == True)
                if(i == 0):
                    return None
                return numpy.vstack((x[0:i], y[0:i])).transpose() # Send path up until collision
        # If no part of the path is in collision
        return numpy.vstack((x,y)).transpose() # Send whole path

    def collision_check(self, x, y):
        transform_t = numpy.eye(4)
        transform_t[0][3] = x
        transform_t[1][3] = y
        self.robot.SetTransform(transform_t)
        collision = self.robot.GetEnv().CheckCollision(self.robot, self.table)
        self.robot.SetTransform(numpy.eye(4))
        return collision

    def ValidConfig(self, config):
        if config[0] < lower_limits[0] or config[1] < lower_limits[1]:
            return False
        if config[0] > upper_limits[0] or config[1] > upper_limits[1]:
            return False
        new_pose = numpy.array([[1, 0, 0, config[0]],
                                    [0, 1, 0, config[1]],
                                    [0, 0, 1, 0     ],
                                    [0, 0, 0, 1     ]])
        self.robot.SetTransform(new_pose)
        #if(self.robot.GetEnv().CheckCollision(robo, self.table) == False):
        #    return True
        #else:
        #    return False
        return !(self.robot.GetEnv().CheckCollision(self.robot, self.table))
        
    def ShortenPath(self, path, timeout=5.0):
        
        # 
        # TODO: Implement a function which performs path shortening
        #  on the given path.  Terminate the shortening after the 
        #  given timout (in seconds).
        #
        #make copy so we don't overwrite the original path
        short_path = list(path)

        init_time = time.time()
        while time.time() - init_time < timeout:
            start = random.randint(0, len(short_path)-3)
            end = random.randint(start+1, len(short_path)-1)
            extend = self.Extend(short_path[start], short_path[end])
            if(extend == None or len(extend) == 0):
                continue
            else:
                if compare(extend[len(extend)-1,:], short_path[end]):
                    short_path[start+1:end] = []

        return numpy.array(short_path)

    def GetEllipsoidScale(self, c_max, c_min):
        scale = numpy.array([[1, 0], [0,1]])
        scale[0,0] = c_max / 2
        scale[1,1] = ((c_max**2 - c_min**2)**0.5)/2 

        return scale

    def InitializePlot(self, goal_config):
        self.fig = pl.figure()
        pl.xlim([self.lower_limits[0], self.upper_limits[0]])
        pl.ylim([self.lower_limits[1], self.upper_limits[1]])
        pl.plot(goal_config[0], goal_config[1], 'gx')

        # Show all obstacles in environment
        for b in self.robot.GetEnv().GetBodies():
            if b.GetName() == self.robot.GetName():
                continue
            bb = b.ComputeAABB()
            pl.plot([bb.pos()[0] - bb.extents()[0],
                     bb.pos()[0] + bb.extents()[0],
                     bb.pos()[0] + bb.extents()[0],
                     bb.pos()[0] - bb.extents()[0],
                     bb.pos()[0] - bb.extents()[0]],
                    [bb.pos()[1] - bb.extents()[1],
                     bb.pos()[1] - bb.extents()[1],
                     bb.pos()[1] + bb.extents()[1],
                     bb.pos()[1] + bb.extents()[1],
                     bb.pos()[1] - bb.extents()[1]], 'r')
                    
                     
        pl.ion()
        pl.show()
        
    def PlotEdge(self, sconfig, econfig):
        pl.plot([sconfig[0], econfig[0]],
                [sconfig[1], econfig[1]],
                'k.-', linewidth=2.5)
        pl.draw()
        
