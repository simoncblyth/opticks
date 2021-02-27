#!/usr/bin/env python

import os, numpy as np, glob
import matplotlib.pyplot as plt 

try:
    import pyvista as pv
except ImportError:
    pv = None
pass

def plot3d(pos):
    pl = pv.Plotter()
    pl.add_points(pos, color='#FFFFFF', point_size=2.0 )  
    pl.show_grid()
    cp = pl.show()
    return cp


np.set_printoptions(suppress=True)

dir_ = lambda:os.environ.get("OUTDIR", os.getcwd())
path_ = lambda name:os.path.join(dir_(), name)
load_ = lambda name:np.load(path_(name))

def sdf_sphere(p,sz):
    """ 
    :param p: intersect coordinates array of shape (n,3)
    :param sz: scalar radius of sphere
    :return d: distance to sphere surface, array of shape (n,) : -ve inside 

    https://iquilezles.org/www/articles/distfunctions/distfunctions.htm

    ::

        float sdSphere( vec3 p, float s )
        {
            return length(p)-s;
        }

    """
    assert len(p.shape) == 2 and p.shape[-1] == 3 and p.shape[0] > 0 
    d = np.sqrt(np.sum(p*p, axis=1)) - sz
    return d


def identity( instance_id, primitive_id ):
    """
    :param instance_id: 1-based instance id
    :param primitive_id: 1-based primitive id

    #  unsigned identity = ( instance_id << 16 ) | (primitive_id << 8) | ( buildinput_id << 0 )  ;
    """
    buildinput_id = primitive_id
    return  ( instance_id << 16 ) | (primitive_id << 8) | ( buildinput_id << 0 )

def pick_intersect_pixels( posi, pick_id ):
    sposi = np.where( posi[:,:,3].view(np.uint32) == pick_id )    
    pick = np.zeros( (*posi.shape[:2], 1), dtype=np.float32 )  
    pick[sposi] = 1 
    #pick_posi = posi[sposi]   
    return pick 


class Geo(object):
    def __init__(self, base):
        ias = {}
        for ias_idx, ias_path in enumerate(sorted(glob.glob("%s/ias_*.npy" % base))):
            ias[ias_idx] = load_(ias_path)
        pass
        gas = {}
        for gas_idx, gas_path in enumerate(sorted(glob.glob("%s/gas_*.npy" % base))):
            gas[gas_idx] = load_(gas_path)
        pass

        ias_ins_idx = ias[0][:,0,3].view(np.uint32)  
        ias_gas_idx = ias[0][:,1,3].view(np.uint32)  

        trs = ias[0].copy()
        trs[:,0,3] = 0.   # scrub the identity info 
        trs[:,1,3] = 0.
        trs[:,2,3] = 0.
        trs[:,3,3] = 1.
        itrs = np.linalg.inv(trs)  ## invert all the IAS transforms at once

        self.trs = trs
        self.itrs = itrs
        self.ias = ias
        self.gas = gas
        self.ias_ins_idx = ias_ins_idx
        self.ias_gas_idx = ias_gas_idx

    def radius(self, gas_idx, prim_idx):
        """
        grabbing radius from bbox : sphere specific
        """
        gas = self.gas[gas_idx]
        bbox = gas[prim_idx] 
        radius = bbox[1][0]   
        return radius

    def sdf(self, gas_idx, prim_idx, lpos):
        radius = self.radius(gas_idx, prim_idx)
        d = sdf_sphere(lpos[:,:3], radius )  # sdf : distances to sphere surface 
        return d 


if __name__ == '__main__':
    base = dir_()
    print(base)
    geo = Geo(base)

    posi = load_("posi.npy")
    hposi = posi[posi[:,:,3] != 0 ]  
    iposi = hposi[:,3].view(np.uint32)  

    #plot3d( hposi[:,:3] )
    #pick_id = identity( 500, 1 ) 
    #pick = pick_intersect_pixels(posi, pick_id )

    #print(posi.shape)
    #print(ias_[0].shape)
    #print(posi.shape)

    pxid = posi[:,:,3].view(np.uint32)      # pixel identity 

    instance_id   = ( pxid & 0xffff0000 ) >> 16    # all three _id are 1-based to distinguish from miss at zero
    primitive_id  = ( pxid & 0x0000ff00 ) >> 8 
    buildinput_id = ( pxid & 0x000000ff ) >> 0 

    assert np.all( primitive_id == buildinput_id ) 

    # identities of all intersected pieces of geometry 
    upxid, upxid_counts = np.unique(pxid, return_counts=True) 
    
    ires = np.zeros( (len(upxid), 4), dtype=np.int32 )
    fres = np.zeros( (len(upxid), 4), dtype=np.float32 )

    # loop over all identified pieces of geometry with intersects
    for i in range(1,len(upxid)):
        zid = upxid[i] 
        zid_count = upxid_counts[i]
        assert zid > 0, "must skip misses at i=0"    # hmm this assumes there are some misses   

        zinstance_id = ( zid & 0xffff0000 ) >> 16 
        zprimitive_id  = ( zid & 0x0000ff00 ) >> 8
        zbuildinput_id = ( zid & 0x000000ff ) >> 0
        assert zprimitive_id == zbuildinput_id 

        zinstance_idx = zinstance_id - 1
        zprimitive_idx = zprimitive_id - 1

        tr = geo.trs[zinstance_idx]
        itr = geo.itrs[zinstance_idx]

        gas_idx = geo.ias_gas_idx[zinstance_idx]   # lookup in the IAS the gas_idx for this instance
        ins_idx = geo.ias_ins_idx[zinstance_idx]   # lookup in the IAS the ins_idx for this instance  
        assert ins_idx == zinstance_idx  # check  

        z = np.where(pxid == zid)   

        zpxid = posi[z][:,3].view(np.uint32).copy()
        zposi = posi[z].copy()  
        zposi[:,3] = 1.      # global 3d coords for intersect pixels, ready for transform
        zlpos = np.dot( zposi, itr ) # transform global positions into instance local ones 

        radius = geo.radius(gas_idx, zprimitive_idx )  
        d = geo.sdf(gas_idx, zprimitive_idx, zlpos[:,:3] )  

        print("i:%5d zid:%9d zid_count:%6d ins_idx:%4d gas_idx:%3d  d.min:%10s d.max:%10s radius:%s  "  % ( i, zid, zid_count, ins_idx, gas_idx, d.min(), d.max(), radius ))
        pass
        fres[i] = (d.min(),d.max(), radius,0. )
        ires[i] = ( len(zposi), ins_idx, gas_idx, zprimitive_idx )
    pass
   
    print("ires\n", ires) 
    print("fres\n", fres) 
    abs_dmax = np.max(np.abs(fres[:,:2]))   
    print("abs_dmax:%s" % abs_dmax)
    print(dir_())

if 0:
    pick_plot = False
    fig, axs = plt.subplots(3 if pick_plot else 2 )
    #axs[0].imshow(instance_id, vmin=0, vmax=10)  # big sphere is there, just not visible as too much range
    axs[0].imshow(instance_id) 
    axs[0].set_xlabel("instance_id %s %s " % (instance_id.min(),instance_id.max()))     
    axs[1].imshow(primitive_id) 
    axs[1].set_xlabel("primitive_id %s %s " % (primitive_id.min(),primitive_id.max()))     

    if pick_plot:
        axs[2].imshow(pick) 
    pass
    fig.show()


