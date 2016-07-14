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


if __name__ == '__main__':
    imm = InstancedMergedMesh("$IDPATH/GMergedMesh/1")



