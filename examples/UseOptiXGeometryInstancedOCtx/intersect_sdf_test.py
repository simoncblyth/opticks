#!/usr/bin/env python
"""
intersect_sdf_test.py
======================

::

   cd ~/opticks/examples/UseOptiXGeometryInstancedOCtx
   ipython -i intersect_sdf_test.py


"""
import numpy as np
import os 

NAME = os.path.basename(os.path.abspath("."))    
TMPDIR = os.path.expandvars("/tmp/$USER/opticks/%s" % NAME ) 
load_ = lambda name:np.load( os.path.join(TMPDIR, "%s.npy" % name)) 
 
BOX = 1
SPHERE = 2
GNAME =  {BOX:"Box", SPHERE:"Sphere"}


def sdf_box(p,sz):
    """
    :param p: coordinates array of shape (n,3)
    :param sz: scalar full side length of box
    :return d: distance to box surface, array of shape (n,) : -ve inside 

    https://iquilezles.org/www/articles/distfunctions/distfunctions.htm

    ::

        float sdBox( vec3 p, vec3 b )
        {
            vec3 q = abs(p) - b;
            return length(max(q,0.0)) + min(max(q.x,max(q.y,q.z)),0.0);
        }

    """
    assert len(p.shape) == 2 and p.shape[-1] == 3 and p.shape[0] > 0
    box = np.array( [sz/2., sz/2., sz/2.] ) 
    q = np.abs(p) - box  
    mqz = np.maximum( q, np.zeros([len(q),3]) ) 
    t1 = np.sqrt(np.sum(mqz*mqz, axis=1)) 
    t2 = np.minimum( np.max(q, axis=1) , 0. )  
    d = t1 + t2
    return d 


def sdf_sphere(p,sz):
    """
    :param p: coordinates array of shape (n,3)
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



def sdf( geocode, lpos, sz):
    """
    :param geocode: scalar
    :param lpos: local frame coordinates array of shape (n,3)
    :param sz: scalar
    :return delta: array of distances to geometry surfaces
    """
    if geocode == BOX: 
        delta = sdf_box(lpos, sz)
    elif geocode == SPHERE: 
        delta = sdf_sphere(lpos, sz)
    else:
        assert 0
    pass
    return delta 

class IntersectSDFTest(object):
    def __init__(self, sz, epsilon):
        self.sz = sz
        self.epsilon = epsilon

        out = load_("out")               ## (height,width,4) image pixels
        posi = load_("posi")             ## (height,width,4) pixel intersect position and distance (float4)
        transforms = load_("transforms") ## (num_tran,4,4)

        print("out  %s : (height,width,uchar4) : image pixels  " % repr(out.shape) )
        print("posi %s : (height,width,float4) : pixel intersect position " % repr(posi.shape) )
        print("transforms %s : (num_tran,float4x4) " % repr(transforms.shape) )

        tidx = transforms[:,0,3].view(np.uint32).copy() 
        transforms[:,0,3] = 0.   ## scrub the identity info 
        itransforms = np.linalg.inv(transforms)  ## invert all the transforms at once

        self.tidx = tidx
        self.out = out
        self.posi = posi
        self.transforms = transforms
        self.itransforms = itransforms

    def select_intersect_transforms(self, geocode):
        """
        :param geocode: type of geometry BOX/SPHERE
        :return tpx: 

        1. spx : pixel coordinates of intersects with a geocode using the intersect_identity 
        2. tpx : unique transform indices for the those spx intersect pixels 
        3. tpx_count : how many of each transform, ie intersects onto each piece of geometry  
        """
        posi = self.posi

        pxid = posi[:,:,3].view(np.uint32)
        gc = pxid >> 24 
        ti = pxid & 0xffffff  

        spx = np.where( gc == geocode )  
        tpx,tpx_count = np.unique( ti[spx], return_counts=True)  

        return tpx 

    def get_local_intersects(self, transform_index):
        """
        1. px : pixel coordinates of intersects that landed on geometry instance with that transform
        2. po : intersection 3d coordinates of all those pixel intersects 
        3. lpo : local frame 3d coordinates of pixel intersects
        """
        itransforms = self.itransforms
        itr = itransforms[transform_index-1]  # index is 1-based

        posi = self.posi

        pxid = posi[:,:,3].view(np.uint32)
        gc = pxid >> 24 
        ti = pxid & 0xffffff  

        px = np.where( ti == transform_index ) 

        assert len(np.unique(gc[px]))==1, " should all be same geocode " 
        assert len(np.unique(ti[px]))==1, " should all be same transform_index " 

        po = posi[px].copy()
        po[:,3] = 1.

        # local frame intersect positions 
        lpo = np.dot( po, itr )[:,:3]
        return lpo


    def check(self, geocode):
        """
        For all pixels check the 3d coordinates of the intersect pixels 
        are on the surface of the expected geometries, by transforming coordinates into 
        the local frames and checking sdf values.
        """

        ist = self 
        tpx = ist.select_intersect_transforms(geocode) 

        dt = np.zeros( [len(tpx), 2] )
        for i, transform_index in enumerate(tpx):

            lpos = ist.get_local_intersects(transform_index)
            delta = sdf(geocode, lpos, self.sz)

            dt[i,0] = delta.min()
            dt[i,1] = delta.max()

            print(" %6s transform_index %3d  sdf delta min/max %f %f  delta_shape %s  " % (GNAME[geocode], transform_index, dt[i,0], dt[i,1], str(delta.shape) ))
        pass

        dtmi = dt.min()
        dtmx = dt.max()
        epsilon = self.epsilon 

        print(" %6s sdf dt min/max %f %f  epsilon %f dt_shape %s  " % (GNAME[geocode], dtmi, dtmx, epsilon, str(dt.shape) ))
        assert np.abs(dtmi) < epsilon, ( dtmi, epsilon )
        assert np.abs(dtmx) < epsilon, ( dtmx, epsilon )
        return dt 



if __name__ == '__main__':
    np.set_printoptions(suppress=True)

    ist = IntersectSDFTest(sz=5., epsilon=4e-4)  
    ist.check(BOX)
    ist.check(SPHERE)
    print("TMPDIR %s " % TMPDIR) 

    posi = ist.posi





