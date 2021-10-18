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


FramePhotons vs Photons
---------------------------

Using frame photons is a trick to effectively see results 
from many more photons that have to pay the costs for tranfers etc.. 
Frame photons lodge photons onto a frame of pixels limiting 
the maximumm number of photons to handle. 

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
    #theme = "default"
    theme = "dark"
    #theme = "paraview"
    #theme = "document"
    pv.set_plot_theme(theme)
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


def shorten_bname(bname):
    elem = bname.split("/")
    assert len(elem) == 4
    omat,osur,isur,imat = elem
    return "/".join([omat,osur[:3],isur[:3],imat])


class PH(object):
    """
    Photon wrapper for re-usable photon data handling 
    """
    def __init__(self, p, gs, cf ):

        self.p = p 
        self.gs = gs
        self.cf = cf
        self.topline = os.environ.get("TOPLINE", "CSGOptiXSimulateTest.py:PH")

        outdir = os.path.join(CSGOptiXSimulateTest.FOLD, "figs")
        if not os.path.isdir(outdir):
            os.makedirs(outdir)
        pass
        self.outdir = outdir

        if p.ndim == 3:
            bnd,ids = self.Photons(p) 
        elif p.ndim == 4:
            bnd,ids = self.FramePhotons(p) 
        pass
        self.boundaries(bnd)
        self.identities(ids)

        self.colors = make_colors()
        self.size = np.array([1280, 720])
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
        """
        Transform enabled gensteps:

        * gs[igs,0,3] photons to generate for genstep *igs* 
        * gs[igs,1] local frame center position
        * gs[igs,2:] 4x4 transform  

        """
        gs_numpho = gs.view(np.int32)[:,0,3] 
        gs_centers = np.zeros( (len(gs), 4 ), dtype=np.float32 )
        for igs in range(len(gs)): gs_centers[igs] = np.dot( gs[igs,1], gs[igs,2:] )  

        copyref( locals(), globals(), self, "gs_" ) 
      
    def identities(self, ids):
        uids, uids_counts = np.unique(ids, return_counts=True)    
        iids = ids[ids>0]     
        i_pr = ids >> 16    
        i_in = ids & 0xffff   
        i_global = i_in == 0 
        i_instance = i_in > 0 

        copyref( locals(), globals(), self, "i_" ) 

    def positions_plt(self, sz=1.0):
        """
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
        igs = slice(None) if len(gs_centers) > 1 else 0

        # this will lead to a non-straight-on view
        # because the gs_centers are in the tilted global frame 

        xlim = np.array([gs_centers[:,X].min(), gs_centers[:,X].max()])
        ylim = np.array([gs_centers[:,Z].min(), gs_centers[:,Z].max()])  
        title = [self.topline,]

        fig, ax = plt.subplots(figsize=self.size/100.)  # mpl uses dpi 100
        fig.suptitle("\n".join(title))

        print("positions_plt")
        for idesc,upos in enumerate(ubnd_descending): 
            if len(isel) > 0 and not idesc in isel: continue 
            bname = ubnd_onames[idesc] 
            label = shorten_bname(bname)
            ub = ubnd[upos]
            ub_count = ubnd_counts[upos] 
            color = colors[idesc % len(colors)]   # gives the more frequent boundary the easy_color names 
            print( " %2d : %4d : %6d : %20s : %40s : %s " % (idesc, ub, ub_count, color, bname, label ))            
            if ub==0: continue # for frame photons, empty pixels give zero 

            pos = p[bnd==ub][:,0]
            ax.scatter( pos[:,X], pos[:,Z], label=label, color=color, s=sz )
        pass

        ax.scatter( gs_centers[igs, X], gs_centers[igs,Z], label="gs_center XZ", s=sz )

        ax.set_xlim( xlim )
        ax.set_ylim( ylim )

        ax.set_aspect('equal')
        ax.legend(loc="upper right", markerscale=4)
        fig.show()
        outpath = os.path.join(self.outdir,"positions_plt.png") 
        print(outpath)
        fig.savefig(outpath)

    def positions_pvplt_simple(self):
        size = self.size
        p = self.p

        pos = p[:,0,:3]

        pl = pv.Plotter(window_size=size*2 )  # retina 2x ?
        pl.add_points( pos, color="white" )        
        cp = pl.show()
        return cp

    def positions_pvplt(self):
        """
        * previously always starts really zoomed in, requiring two-finger upping to see the intersects
        * following hint from https://github.com/pyvista/pyvista/issues/863 now set an adhoc zoom factor
 
        Positioning the eye with a simple global frame y-offset causes distortion 
        and apparent untrue overlaps due to the tilt of the geometry.
        Need to apply the central transform to the gaze vector to get straight on view.
 
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
        size = self.size

        zoom = 4./1000.   # why this adhoc value  ?  does it depend on yoffset ?
        yoffset = -1000.    

        pl = pv.Plotter(window_size=size*2 )  # retina 2x ?
        pl.view_xz() 
        pl.camera.ParallelProjectionOn()  
        pl.camera.Zoom(zoom)

        look = peta[0,1,:3]                          # this is fine
        eye = look + np.array([ 0, yoffset, 0 ])     # problematic eye position 
        up = (0,0,1)                                 # will usually be tilted as global up doesnt match local 

        pl.add_text(self.topline)
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
            label = shorten_bname(bname)
            color = colors[idesc % len(colors)]   # gives the more frequent boundary the easy_color names 

            print( " %2d : %4d : %6d : %20s : %40s : %s " % (idesc, ub, ub_count, color, bname, label ))            
            if ub==0: continue # for frame photons, empty pixels give zero 

            pos = p[bnd==ub][:,0,:3]
            pl.add_points( pos, color=color )
        pass
        outpath = os.path.join(self.outdir,"positions_pvplt.png") 
        print(outpath)
        cp = pl.show(screenshot=outpath)
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
    #ph.positions_pvplt()
    #ph.positions_pvplt_simple()




