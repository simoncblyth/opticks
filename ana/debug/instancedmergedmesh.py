#!/usr/bin/env python
#
# Copyright (c) 2019 Opticks Team. All Rights Reserved.
#
# This file is part of Opticks
# (see https://bitbucket.org/simoncblyth/opticks).
#
# Licensed under the Apache License, Version 2.0 (the "License"); 
# you may not use this file except in compliance with the License.  
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software 
# distributed under the License is distributed on an "AS IS" BASIS, 
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
# See the License for the specific language governing permissions and 
# limitations under the License.
#

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



