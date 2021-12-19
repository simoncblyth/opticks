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

* for comparison of two sets of cxs see x4/xxs.sh 

TODO : identity info improvements
------------------------------------

* retaining the sign of the boundary would be helpful also b=0 is swamped when using FramePhotons

FramePhotons vs Photons
---------------------------

Using frame photons is a trick to effectively see results 
from many more photons that have to pay the costs for tranfers etc.. 
Frame photons lodge photons onto a frame of pixels limiting 
the maximumm number of photons to handle. 

pyvista interaction
----------------------

* to zoom out/in : slide two fingers up/down on trackpad. 
* to pan : hold down shift and one finger tap-lock, then move finger around  


plotting a selection of boundaries only, picked by descending frequency index
----------------------------------------------------------------------------------

::

    cx ; ./grab.sh 
    cx ; ./cxs0.sh py 


    cx ; ipython -i tests/CSGOptiXSimulateTest.py   # all boundaries

    ISEL=0,1         ./cxs.sh    # just the 2 most frequent boundaries
    ISEL=0,1,2,3,4   ./cxs.sh 

    ISEL=Hama        ./cxs.sh    # select boundaries via strings in the bndnames
    ISEL=NNVT        ./cxs.sh 
    ISEL=Pyrex       ./cxs.sh 
    ISEL=Pyrex,Water ./cxs.sh 


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
from opticks.ana.fold import Fold

try:
    import matplotlib.pyplot as mp
except ImportError:
    mp = None
pass
#mp=None

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
#pv=None

X,Y,Z = 0,1,2
SIZE = np.array([1280, 720])


def make_colors():
    """
    :return colors: large list of color names with easily recognisable ones first 
    """
    #colors = ["red","green","blue","cyan","magenta","yellow","pink","purple"]
    all_colors = list(hexcolors.keys())
    easy_colors = "red green blue cyan magenta yellow pink".split()
    skip_colors = "bisque beige white aliceblue antiquewhite".split()    # skip colors that look too alike 

    colors = easy_colors 
    for c in all_colors:
        if c in skip_colors: 
            continue
        if not c in colors: 
            colors.append(c) 
        pass
    pass
    return colors

COLORS = make_colors()


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
    if len(elem) == 4:
        omat,osur,isur,imat = elem
        bn = "/".join([omat,osur[:3],isur[:3],imat])
    else:
        bn = bname
    pass 
    return bn


class Photons(object):
    """
    feat contriols how to select positions, eg  via boundary or identity 
    allow plotting of subsets with different colors
    """
    @classmethod
    def SubMock(cls, i, num):
        p = np.zeros([num, 4, 4], dtype=np.float32)  
        offset = i*100
        for j in range(10):
            for k in range(10): 
                idx = j*10+k
                if idx < num:
                    p[idx,0,0] = float(offset+j*10)
                    p[idx,0,1] = 0
                    p[idx,0,2] = float(offset+k*10)
                    p.view(np.int32)[idx,3,3] = i << 16
                pass
            pass
        pass
        return p

    @classmethod
    def Mock(cls):
        """
        Random number of items between 50 and 100 for each of 10 categories 
        """
        aa = []
        for i in range(10):
            aa.append(cls.SubMock(i, np.random.randint(0,100)))
        pass
        return np.concatenate(tuple(aa))

    def __init__(self, pos, cf=None, featname="pid", do_mok=False ):

        p = pos.p
            
        log.info("[Photons p.ndim %d p.shape %s " % (int(p.ndim), str(p.shape)) )
        assert featname in ["pid", "bnd", "ins", "mok"]
        if p.ndim == 3:
            bnd = p[:,2,3].view(np.int32)
            ids = p[:,3,3].view(np.int32) 
        elif p.ndim == 4:
            bnd = p.view(np.int32)[:,:,2,3]
            ids = p.view(np.int32)[:,:,3,3] 
        else:
            log.info("unexpected p.shape %s " % str(p.shape))
        pass
        pid = ids >> 16
        ins = ids & 0xffff   # ridx?    

        log.info("[ Photons.bndfeat ")
        bnd_namedict = {} if cf is None else cf.bndnamedict 
        bndfeat = Feature("bnd", bnd, bnd_namedict)
        log.info("] Photons.bndfeat ")

        log.info("[ Photons.pidfeat ")
        pid_namedict = {} if cf is None else cf.primIdx_meshname_dict()
        log.info(" pid_namedict: %d  " % len(pid_namedict))
        pidfeat = Feature("pid", pid, pid_namedict)
        log.info("] Photons.pidfeat ")

        log.info("[ Photons.insfeat ")
        ins_namedict = {} if cf is None else cf.insnamedict
        log.info(" ins_namedict: %d  " % len(ins_namedict))
        insfeat = Feature("ins", ins, ins_namedict)
        log.info("] Photons.insfeat ")

        if do_mok:
            log.info("[ Photons.mokfeat ")
            mok_namedict = {} if cf is None else cf.moknamedict 
            mokfeat = Feature("mok", pid, mok_namedict)
            log.info("] Photons.mokfeat ")
        else: 
            mokfeat = None
        pass

        if featname=="pid":
            feat = pidfeat
        elif featname == "bnd":
            feat = bndfeat
        elif featname == "ins":
            feat = insfeat
        elif featname == "mok":
            feat = mokfeat
        else:
            feat = None
        pass

        self.cf = cf
        self.p = p 
        self.bnd = bnd
        self.ids = ids
        self.bndfeat = bndfeat
        self.pidfeat = pidfeat
        self.insfeat = insfeat
        self.mokfeat = mokfeat
        self.feat = feat
        log.info("]Photons")

    def __repr__(self):
        return "\n".join([
               "p %s" % str(self.p.shape), 
               ])


class Feature(object):
    """
    Trying to generalize feature handling 
    """
    def __init__(self, name, val, vname={}):
        """
        :param name: string eg "bnd" or "primIdx"
        :param val: large array of integer feature values 
        :param namedict: dict relating feature integers to string names 

        The is an implicit assumption that the number of unique feature values is not enormous,
        for example boundary values or prim identity values.
        """
        uval, ucount = np.unique(val, return_counts=True)

        if len(vname) == 0:
            nn = ["%s%d" % (name,i) for i in uval]
            vname = dict(zip(uval,nn)) 
        pass
        pass 
        idxdesc = np.argsort(ucount)[::-1]  
        # indices of uval and ucount that reorder those arrays into descending count order

        ocount = [ucount[j]       for j in idxdesc]
        ouval  = [uval[j]         for j in idxdesc]

        # vname needs absolutes to get the names 
        onames = [vname[uval[j]]  for j in idxdesc]


        self.name = name
        self.val = val
        self.vname = vname

        self.uval = uval
        self.unum = len(uval) 
        self.ucount = ucount
        self.idxdesc = idxdesc
        self.onames = onames
        self.ocount = ocount
        self.ouval = ouval

        ISEL = os.environ.get("ISEL","")  
        isel = self.parse_ISEL(ISEL, onames) 
        sisel = ",".join(map(str, isel))

        print( "Feature name %s ISEL: [%s] isel: [%s] sisel [%s] " % (name, ISEL, str(isel), sisel))

        self.isel = isel 
        self.sisel = sisel 

    @classmethod
    def parse_ISEL(cls, ISEL, onames):
        """ 
        :param ISEL: comma delimited list of strings or integers 
        :param onames: names ordered in descending frequency order
        :return isels: list of frequency order indices 

        Integers in the ISEL are interpreted as frequency order indices. 

        Strings are interpreted as fragments to look for in the ordered names,
        (which could be boundary names or prim names for example) 
        eg use Hama or NNVT to yield the list of frequency order indices 
        with corresponding names containing those strings. 
        """
        ISELS = list(filter(None,ISEL.split(",")))
        isels = []
        for i in ISELS:
            if i.isnumeric(): 
                isels.append(int(i))
            else:
                for idesc, nam in enumerate(onames):
                    if i in nam: 
                        isels.append(idesc)
                    pass
                pass
            pass
        pass    
        return isels 

    def __call__(self, idesc):
        """
        :param idesc: zero based index less than unum

        for frame photons, empty pixels give zero : so not including 0 in ISEL allows to skip
        if uval==0 and not 0 in isel: continue 

        """
        assert idesc > -1 and idesc < self.unum
        fname = self.onames[idesc]
        uval = self.ouval[idesc] 
        count = self.ocount[idesc] 
        isel = self.isel  

        if fname[0] == "_":
            fname = fname[1:]
        pass
        #label = "%s:%s" % (idesc, fname)
        label = "%s" % (fname)
        label = label.replace("solid","s")
        color = COLORS[idesc % len(COLORS)]  # gives the more frequent boundary the easy_color names 
        msg = " %2d : %4d : %6d : %20s : %40s : %s " % (idesc, uval, count, color, fname, label )
        selector = self.val == uval

        if len(isel) == 0:
            skip = False
        else:
            skip = idesc not in isel
        pass 
        return uval, selector, label, color, skip, msg 

    def __str__(self):
        lines = []
        lines.append(self.desc)  
        for idesc in range(self.unum):
            uval, selector, label, color, skip, msg = self(idesc)
            lines.append(msg)
        pass
        return "\n".join(lines)

    desc = property(lambda self:"ph.%sfeat : %s " % (self.name, str(self.val.shape)))

    def __repr__(self):
        return "\n".join([
            "Feature name %s val %s" % (self.name, str(self.val.shape)),
            "uval %s " % str(self.uval),
            "ucount %s " % str(self.ucount),
            "idxdesc %s " % str(self.idxdesc),
            "onames %s " % " ".join(self.onames),
            "ocount %s " % str(self.ocount),
            "ouval %s " % " ".join(map(str,self.ouval)),
            ])

class Gensteps(object):
    """
    Transform enabled gensteps:

    * gs[igs,0,3] photons to generate for genstep *igs* 
    * gs[igs,1] local frame center position
    * gs[igs,2:] 4x4 transform  

    """
    def __init__(self, genstep, metatran, local=True ):
        gs = genstep
        mtr = metatran
        log.info("gensteps")

        numpho = gs.view(np.int32)[:,0,3] 
        centers = np.zeros( (len(gs), 4 ), dtype=np.float32 )
        for igs in range(len(gs)): centers[igs] = np.dot( gs[igs,1], gs[igs,2:] )  

        centers_local = np.dot( centers, mtr[1] )  # use metatran.v to transform back to local frame
        ugsc = centers_local if local else centers  

        lim = {}
        lim[X] = np.array([ugsc[:,X].min(), ugsc[:,X].max()])
        lim[Y] = np.array([ugsc[:,Y].min(), ugsc[:,Y].max()])  
        lim[Z] = np.array([ugsc[:,Z].min(), ugsc[:,Z].max()])  

        self.gs = gs
        self.mtr = mtr 
        self.numpho = numpho
        self.centers = centers 
        self.centers_local = centers_local
        self.ugsc = ugsc
        self.lim = lim 


class Peta(object):
    """
    Interpret the metadata 
    """
    def __init__(self, peta):
        ix0,ix1,iy0,iy1 = peta[0,0].view(np.int32)
        iz0,iz1,photons_per_genstep,zero = peta[0,1].view(np.int32)
        ce = tuple(peta[0,2])
        sce = ("%7.2f" * 4 ) % ce

        assert photons_per_genstep > 0
        assert zero == 0 
        nx = (ix1 - ix0)//2 
        ny = (iy1 - iy0)//2  
        nz = (iz1 - iz0)//2  

        nx_over_nz = float(nx)/float(nz)
        if nx_over_nz > 1.:
            axes = X,Z 
        else:
            axes = Z,X
        pass
        print("axes %s " % str(axes))
        # expecting 2D with no action in ny

        log.info(" ix0 %d ix1 %d nx %d  " % (ix0, ix1, nx)) 
        log.info(" iy0 %d iy1 %d ny %d  " % (iy0, iy1, ny)) 
        log.info(" iz0 %d iz1 %d nz %d  " % (iz0, iz1, nz)) 
        log.info(" nx_over_nz %s axes %s X %d Y %d Z %d" % (nx_over_nz,str(axes), X,Y,Z) )

        self.peta = peta
        self.ce = ce
        self.sce = sce
        self.thirdline = " ce: " + sce 

        self.nx_over_nz = nx_over_nz   
        self.axes = axes 
 
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.photons_per_genstep = photons_per_genstep
 

class Positions(object):
    """
    Transforms global intersect positions into local frame 
    """
    def __init__(self, p, gs, peta, local=True, pos_mask=False ):

        mtr = gs.mtr                    # transform

        gpos = p[:,0].copy()            # global frame intersect positions
        gpos[:,3] = 1  
        lpos = np.dot( gpos, mtr[1] )   # local frame intersect positions
        upos = lpos if local else gpos

        poslim = {}
        poslim[X] = np.array([upos[:,X].min(), upos[:,X].max()])
        poslim[Y] = np.array([upos[:,Y].min(), upos[:,Y].max()])  
        poslim[Z] = np.array([upos[:,Z].min(), upos[:,Z].max()])  

        self.poslim = poslim 
        self.gs = gs
        self.peta = peta 

        self.p = p 
        self.upos = upos
        self.local = local

        #self.make_histogram()

        if pos_mask:
            self.apply_pos_mask()
        pass

    def apply_pos_mask(self):
        lim = self.gs.lim  

        xmin, xmax = lim[0] 
        ymin, ymax = lim[1] 
        zmin, zmax = lim[2] 

        upos = self.upos
        xmask = np.logical_and( upos[:,0] >= xmin, upos[:,0] <= xmax )
        ymask = np.logical_and( upos[:,1] >= ymin, upos[:,1] <= ymax )
        zmask = np.logical_and( upos[:,2] >= zmin, upos[:,2] <= zmax )
        xz_mask = np.logical_and( xmask, zmask )
        mask = xz_mask 

        self.mask = mask
        self.p = self.p[mask]
        self.upos = upos[mask]


    def make_histogram(self):
        lim = self.gs.lim  
        nx = self.peta.nx
        ny = self.peta.ny
        nz = self.peta.nz
        upos = self.upos

        bins = (np.linspace(*lim[X], 2*nx+1), np.linspace(*lim[Y], max(2*ny+1,2) ), np.linspace(*lim[Z], 2*nz+2))
        h3d, bins2 = np.histogramdd(upos[:,:3], bins=bins )   
        ## TODO: use the 3d histo to sparse-ify gensteps positions, to avoiding shooting rays from big voids 

        self.h3d = h3d
        self.bins = bins 
        self.bins2 = bins2 

    def pvplt_simple(self):
        p = self.p
        pos = p[:,0,:3]

        pl = pv.Plotter(window_size=SIZE*2 )  # retina 2x ?
        pl.add_points( pos, color="white" )        
        cp = pl.show()
        return cp


class Plt(object):
    def __init__(self, outdir, feat, gs, peta, pos ):

        self.outdir = outdir 
        self.feat = feat
        self.gs = gs
        self.peta = peta
        self.pos = pos

        self.topline = os.environ.get("TOPLINE", "CSGOptiXSimulateTest.py:PH")
        self.botline = os.environ.get("BOTLINE", "cxs") 

        efloatlist_ = lambda ekey:list(map(float, filter(None, os.environ.get(ekey,"").split(","))))
        self.xx = efloatlist_("XX")
        self.yy = efloatlist_("YY")
        self.zz = efloatlist_("ZZ")
        self.sz = float(os.environ.get("SZ","1.0"))
        self.zoom = float(os.environ.get("ZOOM","3.0"))
        self.look = np.array( list(map(float, os.environ.get("LOOK","0.,0.,0.").split(","))) )
 
    def outpath_(self, stem="positions", ptype="pvplt"):
        sisel = self.feat.sisel
        return os.path.join(self.outdir,"%s_%s_%s.png" % (stem, ptype, self.feat.name)) 

    def positions_mpplt(self, legend=True):
        """
        (H,V) are the plotting axes 
        (X,Y,Z) = (0,1,2) correspond to absolute axes which can be mapped to plotting axes in various ways 
        
        when Z is vertical lines of constant Z appear horizontal 
        when Z is horizontal lines of constant Z appear vertical 

        """
        upos = self.pos.upos
        ugsc = self.gs.ugsc
        lim = self.gs.lim
        H,V = self.peta.axes      ## traditionally H,V = X,Z  but are now generalizing 
        feat = self.feat
        sz = self.sz
        print("positions_mpplt feat.name %s " % feat.name )

        xlim = lim[X]
        ylim = lim[Y]
        zlim = lim[Z]

        igs = slice(None) if len(ugsc) > 1 else 0

        title = [self.topline, self.botline, self.peta.thirdline]

        fig, ax = mp.subplots(figsize=SIZE/100.)  # mpl uses dpi 100
        fig.suptitle("\n".join(title))

        for idesc in range(feat.unum):
            uval, selector, label, color, skip, msg = feat(idesc)
            if skip: continue
            pos = upos[selector] 
            ax.scatter( pos[:,H], pos[:,V], label=label, color=color, s=sz )
        pass

        log.info(" xlim[0] %8.4f xlim[1] %8.4f " % (xlim[0], xlim[1]) )
        log.info(" ylim[0] %8.4f ylim[1] %8.4f " % (ylim[0], ylim[1]) )
        log.info(" zlim[0] %8.4f zlim[1] %8.4f " % (zlim[0], zlim[1]) )

        if H == X and V == Z:   
            for z in self.zz:   # ZZ horizontals 
                label = "z:%8.4f" % z
                ax.plot( lim[H], [z,z], label=None )
            pass
            for x in self.xx:    # XX verticals 
                label = "x:%8.4f" % x
                ax.plot( [x, x], lim[V], label=None ) 
            pass
        elif H == Z and V == X:  ## ZZ verticals 
            for z in self.zz:   
                label = "z:%8.4f" % z
                ax.plot( [z, z], lim[V], label=None )
            pass
            for x in self.xx:    # XX horizontals
                label = "x:%8.4f" % x
                ax.plot( lim[H], [x, x], label=None ) 
            pass
        pass

        label = "gs_center XZ"
        ax.scatter( ugsc[igs, H], ugsc[igs,V], label=None, s=sz )

        ax.set_xlim( lim[H] )
        ax.set_ylim( lim[V] )  # zlim -> ylim 

        ax.set_aspect('equal')
        if legend:
            ax.legend(loc="upper right", markerscale=4)
            ptype = "mpplt"  
        else:
            ptype = "mpnky"  
        pass
        fig.show()

        outpath = self.outpath_("positions",ptype )
        print(outpath)
        fig.savefig(outpath)

    def positions_pvplt(self):
        """
        * actually better to use set_position reset=True after adding points to auto get into ballpark 

        * previously always starts really zoomed in, requiring two-finger upping to see the intersects
        * following hint from https://github.com/pyvista/pyvista/issues/863 now set an adhoc zoom factor
 
        Positioning the eye with a simple global frame y-offset causes distortion 
        and apparent untrue overlaps due to the tilt of the geometry.
        Need to apply the central transform to the gaze vector to get straight on view.

        https://docs.pyvista.org/api/core/_autosummary/pyvista.Camera.zoom.html?highlight=zoom

        In perspective mode, decrease the view angle by the specified factor.

        In parallel mode, decrease the parallel scale by the specified factor.
        A value greater than 1 is a zoom-in, a value less than 1 is a zoom-out.       

        """
        lim = self.gs.lim
        ugsc = self.gs.ugsc

        xlim = lim[X]
        ylim = lim[Y]
        zlim = lim[Z]

        H,V = self.peta.axes      ## traditionally H,V = X,Z  but are now generalizing 
 
        upos = self.pos.upos
        grid = True if self.peta.ny == 0 else False   # grid is too obscuring with 3D

        feat = self.feat 
        zoom = self.zoom
        look = self.look if self.pos.local else self.peta.peta[0,1,:3]

        yoffset = -1000.       ## with parallel projection are rather insensitive to eye position distance
        eye = look + np.array([ 0, yoffset, 0 ])    

        pl = pv.Plotter(window_size=SIZE*2 )  # retina 2x ?
        self.pl = pl 

        if H == X and V == Z:
            up = (0,0,1)                               
        elif H == Z and V == X:
            up = (-1,0,0)                               
        else:
            assert 0
        pass
        pl.view_xz()   ## TODO: see if view_xz is doing anything when subsequently set_focus/viewup/position 

        pl.camera.ParallelProjectionOn()  
        pl.add_text(self.topline, position="upper_left")
        pl.add_text(self.botline, position="lower_left")
        pl.add_text(self.peta.thirdline, position="lower_right")
        print("positions_pvplt feat.name %s " % feat.name )

        for idesc in range(feat.unum):
            uval, selector, label, color, skip, msg = feat(idesc)
            if skip: continue
            pos = upos[selector] 
            print(msg)
            pl.add_points( pos[:,:3], color=color )
        pass

        grid = True 
        if grid:
            pl.add_points( ugsc[:,:3], color="white" )   # genstep grid
        pass   

        for z in self.zz:  # ZZ horizontals (when using traditional XZ axes)
            xhi = np.array( [xlim[1], 0, z] )  # RHS
            xlo = np.array( [xlim[0], 0, z] )  # LHS
            line = pv.Line(xlo, xhi)
            pl.add_mesh(line, color="w")
        pass
        for x in self.xx:    # XX verticals (when using traditional XZ axes)
            zhi = np.array( [x, 0, zlim[1]] )  # TOP
            zlo = np.array( [x, 0, zlim[0]] )  # BOT
            line = pv.Line(zlo, zhi)
            pl.add_mesh(line, color="w")
        pass

        pl.set_focus(    look )
        pl.set_viewup(   up )
        pl.set_position( eye, reset=True )   ## for reset=True to succeed to auto-set the view, must do this after add_points etc.. 
        pl.camera.Zoom(2)

        outpath = self.outpath_("positions","pvplt")
        print(outpath)
        cp = pl.show(screenshot=outpath)
        return cp


def test_mok(cf):
    mock_photons = Photons.Mock()
    ph = Photons(mock_photons, cf, featname="mok", do_mok=True)
    print(ph.mokfeat)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    GEOM = os.environ.get("GEOM", None) ; assert not GEOM is None

    FOLD_DEFAULT = os.path.expandvars("/tmp/$USER/opticks/GeoChain/$GEOM" )

    FOLD = os.environ.get("FOLD", FOLD_DEFAULT ) ## override FOLD set in cxs.sh 

    log.info("FOLD %s" % FOLD )

    cf = CSGFoundry(os.path.join(FOLD, "CSGFoundry"))
    #test_mok(cf)

    cxs = Fold.Load( FOLD, "CSGOptiXSimulateTest", GEOM, globals=True ) 

    outdir = os.path.join(cxs.base, "figs")
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    pass

    gs = Gensteps(cxs.genstep, cxs.metatran)

    peta = Peta(cxs.peta)

    pos = Positions(cxs.photons, gs, peta, local=True, pos_mask=True )

    #pos.pvplt_simple()

    ## TODO: use pos.mask to make legend items more pertinent by only incluing entries for visible intersects

    ph = Photons(pos, cf, featname="pid" )  # pid:meshname, bnd:boundary, ins:instance
    print(ph.bndfeat)
    print(ph.pidfeat)
    print(ph.insfeat)
    feat = ph.feat 

    plt = Plt(outdir, feat, gs, peta, pos )

    if not mp is None:
        plt.positions_mpplt(legend=True)
        #plt.positions_mpplt(legend=False)   # when not using pos_mask legend often too big, so can switch it off 
    pass

    if not pv is None:
        plt.positions_pvplt()
    pass

