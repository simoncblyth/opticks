#!/usr/bin/env python
"""
This demonstrates that volume_meshes.npy can be eliminated, as it matches volume_identity[:,1] for all mm
"""
import os, numpy as np

m = np.load("volume_meshes.npy")
i = np.load("volume_identity.npy")
n = np.load("volume_nodeinfo.npy")

print("volume_meshes mesh_index\n",m)
print("volume_identity node_index/mesh_index/boundary/identity_index(copyNumber) GVolume::getIdentity\n",i)
print("volume_nodeinfo num_face/num_vert/node_index/parent_index\n",n)

assert np.all( i[:,1] ==  m[:,0] ) 

