import operator
import numpy

class RRTTree(object):
    
    def __init__(self, planning_env, start_config):
        
        self.planning_env = planning_env
        self.vertices = dict()
        vid = self.planning_env.discrete_env.ConfigurationToNodeId(start_config)
        self.vertices[vid] = [] # Changed to a list of edges
        self.edges = [] # list of tuples that create edges
        self.start_config=start_config

    def GetRootId(self):
        return 0

    def GetKNearestVertex(self, config, K):
        
        dists = dict()
        for index in self.vertices:
            dists[index] = self.planning_env.ComputeDistance(config, self.vertices[index])

        vids = []
        vdists = []
        for i in xrange(0, K):
            vid, vdist = min(dists.items(), key=lambda x: x[1])
            vids.append(vid)
            vdists.append(vdist)
            del dists[vid]
            if(len(dists) == 0):
                break

        vertices = []
        for vid in vids:
            vertices.append(self.vertices[vid])
        return vids, vertices

    def GetNearestVertex(self, config):
        dists = dict()
        for index in self.vertices.keys():
            if index==self.planning_env.discrete_env.ConfigurationToNodeId(self.start_config):
                dists[index]=999
                pass
            dists[index] = self.planning_env.ComputeDistance(self.planning_env.discrete_env.ConfigurationToNodeId(config), index)

        # vid, vdist = min(dists.items(), key=operator.itemgetter(0))
        vid = min(dists, key=dists.get)

        return vid, self.vertices[vid]

    def AddVertex(self, config):
        vid = self.planning_env.discrete_env.ConfigurationToNodeId(config)
        self.vertices[vid] = []
        return vid

    def AddEdge(self, sid, eid):
        if (sid, eid) not in self.edges:
            self.edges.append((sid,eid))
        self.vertices[sid].append(eid)
        self.vertices[eid].append(sid)

