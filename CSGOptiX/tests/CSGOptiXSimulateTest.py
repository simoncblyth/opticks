#!/usr/bin/env python
"""
tests/CSGOptiXSimulateTest.py
==============================

This is useful as it allows interactive visualization of workstation 
generated intersect data fphoton.npy on remote machines such as 
user laptops that support pyvista. 
 
But that raises the strong possibility that the geocache+CSGFoundry 
used on the laptop does not match that used on the workstation to generate
the fphoton.npy intersects

* this will cause incorrect identity information for intersects. 
* the problem is especially acute when multiple geometries are in use
  and geometries are changed frequently 


pyvista interaction
----------------------

The plot tends to start in a very zoomed in position. 

* to zoom out/in : slide two fingers up/down on trackpad. 
* to pan : hold down shift and one finger tap-lock, then move finger around  


TODO : identity info improvements
------------------------------------

* retaining the sign of the boundary would be helpful also b=0 is swamped


plotting a selection of boundaries only, picked by descending frequency index
----------------------------------------------------------------------------------

::

    cx ; ./grab.sh 
    cx ; ./cxs0.sh py 


    cx ; ipython -i tests/CSGOptiXSimulateTest.py   # all boundaries

    ISEL=0,1         ipython -i tests/CSGOptiXSimulateTest.py    # just the 2 most frequent boundaries
    ISEL=0,1,2,3,4   ipython -i tests/CSGOptiXSimulateTest.py 

    ISEL=Hama        ipython -i tests/CSGOptiXSimulateTest.py    # select boundaries via strings in the bndnames
    ISEL=NNVT        ipython -i tests/CSGOptiXSimulateTest.py 
    ISEL=Pyrex       ipython -i tests/CSGOptiXSimulateTest.py 
    ISEL=Pyrex,Water ipython -i tests/CSGOptiXSimulateTest.py 


AVOIDING INCORRECT IDENTITY INFO ?
------------------------------------

* digest info json metadata accompanying the intersects and foundry would allows the 
  situation to be detected.  

* transfering the CSGFoundry folder from workstation, so the identitity backing information is 
  guaranteed to be consistent with the intersects even as geometry is frequently changed...
  with this in mind think about the size of the CSGFoundry/geocache what can be skipped 
  and what is essential for identity info

::

    cg ; ./grab.sh 



FOUNDRY IDENTITY INFO
------------------------

Currently the only names CSGFoundry holds are mesh names


__closesthit__ch::

    331     unsigned instance_idx = optixGetInstanceId() ;    // see IAS_Builder::Build and InstanceId.h 
    332     unsigned prim_idx  = optixGetPrimitiveIndex() ;  // see GAS_Builder::MakeCustomPrimitivesBI_11N  (1+index-of-CSGPrim within CSGSolid/GAS)
    333     unsigned identity = (( prim_idx & 0xffff ) << 16 ) | ( instance_idx & 0xffff ) ;

    prim_idx = ( i >> 16 )      ## index of bbox within within the GAS 
    instance_idx = i & 0xffff   ## flat 

NB getting zero for the flat instance_idx (single IAS, all transforms in it) 
**DOES** tell you that its a global intersect 

Now how to lookup what a prim_id corresponds to ?

"""
import os, logging, numpy as np
log = logging.getLogger(__name__)
np.set_printoptions(suppress=True, edgeitems=5, linewidth=200,precision=3)
from opticks.CSG.CSGFoundry import CSGFoundry 
import matplotlib.pyplot as plt

try:
    import pyvista as pv
    from pyvista.plotting.colors import hexcolors  
except ImportError:
    pv = None
    hexcolors = None
pass

class CSGOptiXSimulateTest(object):
    CXS = os.environ.get("CXS", "1")
    FOLD = os.path.expandvars("/tmp/$USER/opticks/CSGOptiX/CSGOptiXSimulateTest/%s" % CXS)
    def __init__(self, fold=FOLD):
        print("CXS : %s : loading from fold : %s " % (self.CXS,fold) )
        names = os.listdir(fold)
        for name in filter(lambda n:n.endswith(".npy"),names):
            path = os.path.join(fold, name)
            stem = name[:-4]
            a = np.load(path)
            print(" %10s : %15s : %s " % (stem, str(a.shape), path )) 
            globals()[stem] = a
        pass


def make_colors():
    """
    :return colors: large list of color names with easily recognisable ones first 
    """
    #colors = ["red","green","blue","cyan","magenta","yellow","pink","purple"]
    all_colors = list(hexcolors.keys())
    easy_colors = "red green blue cyan magenta yellow pink".split()
    skip_colors = "aliceblue".split()    # skip colors that look too alike 

    colors = easy_colors 
    for c in all_colors:
        if c in skip_colors: 
            continue
        if not c in colors: 
            colors.append(c) 
        pass
    pass
    return colors


def parse_isel():
    isel = list(map(int, list(filter(None,os.environ.get("ISEL", "").split(","))) ))
    return isel 


def copyref( l, g, s, kps ):
    """
    Copy selected references between scopes::
        
        copyref( locals(), globals(), self, "bnd,ubnd" )

    :param l: locals() 
    :param g: globals() or None
    :param s: self or None
    :param kps: space delimited string identifying quantities to be copied

    The advantage with using this is that can benefit from developing 
    fresh code directly into classes whilst broadcasting the locals from the 
    classes into globals for easy debugging. 
    """
    for k,v in l.items():
        kmatch = np.any(np.array([k.startswith(kp) for kp in kps.split()]))
        if kmatch:
            if not g is None: g[k] = v
            if not s is None: setattr(s, k, v )
            print(k)
        pass
    pass


def fromself( l, s, kk ):
    # nope seems cannot update locals() like this
    # possibly generate some simple code and eval it is a workaround 
    for k in kk.split(): 
        log.info(k)
        l[k] = getattr(s, k)
    pass


class PH(object):
    """
    Photon wrapper for re-usable photon data handling 
    """
    def __init__(self, p, gs, cf ):

        self.p = p 
        self.gs = gs
        self.cf = cf

        if p.ndim == 3:
            bnd,ids = self.Photons(p) 
        elif p.ndim == 4:
            bnd,ids = self.FramePhotons(p) 
        pass
        self.boundaries(bnd)
        self.identities(ids)

        colors = make_colors()
        self.colors = colors
        self.gensteps(gs)

    @classmethod
    def Photons(cls, p):
        log.info("Photons : %s " % str(p.shape)) 
        assert p.ndim == 3
        bnd = p[:,2,3].view(np.int32)
        ids = p[:,3,3].view(np.int32) 
        return bnd, ids

    @classmethod
    def FramePhotons(cls, p):
        log.info("FramePhotons : %s " % str(p.shape)) 
        assert p.ndim == 4
        bnd = p.view(np.int32)[:,:,2,3]
        ids = p.view(np.int32)[:,:,3,3] 
        return bnd, ids

    def boundaries(self, bnd):
        cf = self.cf
        ubnd, ubnd_counts = np.unique(bnd, return_counts=True) 
        ubnd_names = [cf.bndname[b] for b in ubnd]
        ubnd_descending = np.argsort(ubnd_counts)[::-1]
        ubnd_onames = [ubnd_names[j] for j in ubnd_descending]
        isel = cf.parse_ISEL(os.environ.get("ISEL",""), ubnd_onames) 
        print( "isel: %s " % str(isel))
        copyref( locals(), globals(), self, "bnd ubnd isel" ) 

    def gensteps(self, gs):
        gs_numpho = gs.view(np.int32)[:,0,3] 
        gs_centers = gs[:,1,:3] 
        copyref( locals(), globals(), self, "gs_" ) 
      
    def identities(self, ids):
        uids, uids_counts = np.unique(ids, return_counts=True)    
        iids = ids[ids>0]     
        i_pr = ids >> 16    
        i_in = ids & 0xffff   
        i_global = i_in == 0 
        i_instance = i_in > 0 

        copyref( locals(), globals(), self, "i_" ) 

    def positions_plt(self, igs=0, centered=False):
        """
        For simplicity this plotting currently handles a single genstep only
        TODO: expand to multiple genstep perhaps using igs slice(None)
        """
        p = self.p
        gs_centers = self.gs_centers
        bnd = self.bnd
        ubnd = self.ubnd
        ubnd_descending = self.ubnd_descending
        ubnd_onames = self.ubnd_onames
        isel = self.isel
        colors = self.colors

        X,Y,Z = 0,1,2

        fig, ax = plt.subplots(figsize=[12.8, 7.2])

        print("positions_plt")
        for idesc,upos in enumerate(ubnd_descending): 
            if len(isel) > 0 and not idesc in isel: continue 
            bname = ubnd_onames[idesc] 
            ub = ubnd[upos]
            ub_count = ubnd_counts[upos] 
            color = colors[idesc % len(colors)]   # gives the more frequent boundary the easy_color names 
            print( " %2d : %4d : %6d : %20s : %40s " % (idesc, ub, ub_count, color, bname ))            
            if ub==0: continue # for frame photons, empty pixels give zero 

            pos = p[bnd==ub][:,0]
            if centered:
                pos -= gs_centers[igs]
            pass
            ax.scatter( pos[:,X], pos[:,Z], label="xz %d:%s " % (ub, bname), color=color )
        pass

        if not centered:
            ax.scatter( gs_centers[igs, X], gs_centers[igs,Z], label="gs_center XZ" )
        pass

        ax.set_aspect('equal')
        ax.legend()
        fig.show()

    def positions_pvplt(self):
        """
        * always starts really zoomed in, requiring two-finger upping to see the intersects
        """
        p = self.p
        colors = self.colors
        bnd = self.bnd
        ubnd = self.ubnd
        ubnd_descending = self.ubnd_descending
        ubnd_counts = self.ubnd_counts
        ubnd_onames = self.ubnd_onames
        gs_centers = self.gs_centers
        isel = self.isel

        size = np.array( [1024, 768] )*2

        pl = pv.Plotter(window_size=size )
        pl.view_xz() 
        pl.camera.ParallelProjectionOn()  

        look = peta[0,1,:3]  
        eye = look + np.array([ 0, -100, 0 ])  
        up = (0,0,1)

        pl.set_position( eye, reset=False )
        pl.set_focus(    look )
        pl.set_viewup(   up )

        pl.add_points( gs_centers[:,:3], color="white" )           # genstep grid

        print("positions_pvplt")
        for idesc,upos in enumerate(ubnd_descending): 
            if len(isel) > 0 and not idesc in isel: continue 

            ub = ubnd[upos]
            ub_count = ubnd_counts[upos] 
            bname = ubnd_onames[idesc]
            color = colors[idesc % len(colors)]   # gives the more frequent boundary the easy_color names 

            print( " %2d : %4d : %6d : %20s : %40s " % (idesc, ub, ub_count, color, bname ))            
            if ub==0: continue # for frame photons, empty pixels give zero 

            pos = p[bnd==ub][:,0,:3]
            pl.add_points( pos, color=color )
        pass
        #pl.show_grid()
        cp = pl.show()
        return cp


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    cf = CSGFoundry()
    cxs = CSGOptiXSimulateTest()

    g = genstep
    p = photons
    f = fphoton

    #p_or_f = f 
    p_or_f = p 

    ph = PH(p_or_f, genstep, cf)
    ph.positions_plt()
    ph.positions_pvplt()

