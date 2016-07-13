#!/usr/bin/env python
"""
Split off instanced merged mesh specifics into InstancedMergedMesh
to keep MergedMesh simple.
"""

from opticks.ana.mergedmesh import MergedMesh

class InstancedMergedMesh(MergedMesh):
    def __init__(self, base, node_offset=0):
        MergedMesh.__init__(self, base)
        self.node_offset = node_offset

    def vertices_(self, i):
        """
        nodeinfo[:,1] provides the vertex counts, to convert a node index
        into a vertex range potentially with a node numbering offset need
        to add up all vertices prior to the one wish to access 

        ::

            In [6]: imm.nodeinfo
            Out[6]: 
            array([[ 720,  362, 3199, 3155],
                   [ 672,  338, 3200, 3199],
                   [ 960,  482, 3201, 3200],
                   [ 480,  242, 3202, 3200],
                   [  96,   50, 3203, 3200]], dtype=uint32)


        """
        no = self.node_offset
        v_count = self.nodeinfo[0:no+i, 1]  # even when wish to ignore a solid, still have to offset vertices 
        v_offset = v_count.sum()
        v_number = self.nodeinfo[no+i, 1]
        log.info("no %s v_count %s v_number %s v_offset %s " % (no, repr(v_count), v_number, v_offset))
        return self.vertices[v_offset:v_offset+v_number] 

    def rz_(self, i):
        v = self.vertices_(i)
        r = np.linalg.norm(v[:,:2], 2, 1)

        rz = np.zeros((len(v),2), dtype=np.float32)
        rz[:,0] = r*np.sign(v[:,0])
        rz[:,1] = v[:,2]
        return rz 


if __name__ == '__main__':
    imm = InstancedMergedMesh("$IDPATH/GMergedMesh/1")



