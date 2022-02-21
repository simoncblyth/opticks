#!/usr/bin/env python
"""
CSGIntersectSolidTest.py
==========================


AnnulusFourBoxUnion
    notice that spurious intersects all from 2nd circle roots  
 

IXYZ=-6,0,0 SPUR=1 ./csg_geochain.sh ana



"""
import os, logging, numpy as np
from opticks.ana.fold import Fold
from opticks.sysrap.SCenterExtentGenstep import SCenterExtentGenstep

from opticks.ana.gridspec import GridSpec, X, Y, Z
from opticks.ana.npmeta import NPMeta
from opticks.ana.eget import efloat_, efloatlist_, eint_, eintlist_


SIZE = np.array([1280, 720])

log = logging.getLogger(__name__)
np.set_printoptions(suppress=True, edgeitems=5, linewidth=200,precision=3)

try:
    import matplotlib.pyplot as mp  
except ImportError:
    mp = None
pass
#mp = None

try:
    import vtk
    import pyvista as pv
    from pyvista.plotting.colors import hexcolors  
    themes = ["default", "dark", "paraview", "document" ]
    pv.set_plot_theme(themes[1])
except ImportError:
    pv = None
    hexcolors = None
pass
#pv = None



def make_arrows(cent, direction, mag=1):
    direction = direction.copy()
    if cent.ndim != 2:
        cent = cent.reshape((-1, 3))

    if direction.ndim != 2:
        direction = direction.reshape((-1, 3))

    direction[:,0] *= mag
    direction[:,1] *= mag
    direction[:,2] *= mag

    pdata = pv.vector_poly_data(cent, direction)
    # Create arrow object
    arrow = vtk.vtkArrowSource()
    arrow.Update()
    glyph3D = vtk.vtkGlyph3D()
    glyph3D.SetSourceData(arrow.GetOutput())
    glyph3D.SetInputData(pdata)
    glyph3D.SetVectorModeToUseVector()
    glyph3D.Update()

    arrows = pv.wrap(glyph3D.GetOutput())
    return arrows



def lines_rectangle_YX(center, halfside):
    """

          1                0
           +------+------+
           |      |      |
           |      |      |
           |- - - + - - -| 
           |      |      |
           |      |      |
           +------+------+
         2                3


         X
         |
         +-- Y

    """
    p0 = np.array( [center[0]+halfside[0], center[1]+halfside[1], center[2] ])
    p1 = np.array( [center[0]+halfside[0], center[1]-halfside[1], center[2] ])
    p2 = np.array( [center[0]-halfside[0], center[1]-halfside[1], center[2] ])
    p3 = np.array( [center[0]-halfside[0], center[1]+halfside[1], center[2] ])

    ll = np.zeros( (4,2,3),  dtype=np.float32 )

    ll[0] = p0, p1
    ll[1] = p1, p2
    ll[2] = p2, p3
    ll[3] = p3, p0

    return ll.reshape(-1,3)


def AnnulusFourBoxUnion_YX(pl, opt="+X -X +Y -Y circ", color='cyan', width=1, radius=45.):
    if opt.find("circ") > -1:
        arc = pv.CircularArc([0,  radius, 0], [0, radius, 0], [0, 0, 0], negative=True)
        pl.add_mesh(arc, color='cyan', line_width=1)
    pass
    lls = [] 
    if opt.find("+X") > -1:
        llpx = lines_rectangle_YX([ 52., 0.,0.],[10.,11.5, 6.5]) 
        lls.append(llpx)
    pass
    if opt.find("-X") > -1:   
        llnx = lines_rectangle_YX([-52., 0.,0.],[10.,11.5, 6.5]) 
        lls.append(llnx)
    pass
    if opt.find("+Y") > -1:
        llpy = lines_rectangle_YX([0., 50.,0.],[15.,15.,  6.5])
        lls.append(llpy)
    pass
    if opt.find("-Y") > -1:
        llny = lines_rectangle_YX([0.,-50.,0.],[15.,15.,  6.5])
        lls.append(llny)
    pass
    for ll in lls:
        pl.add_lines(ll, color=color, width=width )
    pass

def ReferenceGeometry(pl):
    geom = os.environ.get("GEOM",None)
    if geom is None: return 

    print("ReferenceGeometry for GEOM %s " % geom)
    if geom == "AnnulusFourBoxUnion_YX":
        AnnulusFourBoxUnion_YX(pl, opt="+X -X +Y -Y circ")
    elif geom == "BoxCrossTwoBoxUnion_YX":
        AnnulusFourBoxUnion_YX(pl, opt="+Y +X")
    

    else:
        print("ReferenceGeometry not implemented for GEOM %s " % geom)
    pass



class Rays(object):
    """

    ## different presentation of selected intersects 
    ## red position points (usually invisible under arrow) and normal arrows    

    #pl.add_arrows( s_ray_origin, s_ray_direction, color="blue", mag=s_t ) 
    # drawing arrow from ray origins to intersects works, but the arrows are too big 
    # TODO: find way to do this with different presentation
    # as it avoids having to python loop over the intersects as done
    # below which restricts to small numbers  
    """
    @classmethod
    def Draw(cls, pl, p0, p1, n1, n0=None, ori_color="magenta", nrm_color="red", ray_color="blue", ray_width=1, pos_shift=None ):
        """
        :param pl: pyvista plotter
        :param p0: start position
        :param p1: end position 
        :param n1: direction vec at end
        :param n0: direction vec at start or None
        :param pos_shift: shift vector or None
        """
        assert len(p0) == len(p1) 

        if not pos_shift is None:
            q0 = p0 + pos_shift
            q1 = p1 + pos_shift
        else:
            q0 = p0 
            q1 = p1 
        pass

        pl.add_points( q0, color=nrm_color )   

        if not n1 is None:
            assert len(q0) == len(n1) 
            pl.add_arrows( q1, n1, color=nrm_color, mag=10 )
        pass

        if not n0 is None:
            assert len(q0) == len(n0)
            pl.add_arrows( q0, n0, color=nrm_color, mag=10 )
        pass

        ll = np.zeros( (len(q0), 2, 3), dtype=np.float32 )
        ll[:,0] = q0
        ll[:,1] = q1

        pl.add_points( q0, color=ori_color, point_size=16.0 )
        for i in range(len(ll)):
            pl.add_lines( ll[i].reshape(-1,3), color=ray_color, width=ray_width )
        pass  
    pass


class CSGRecord(object):
    @classmethod
    def Draw0(cls, pl, recs): 
        """
        # works but then difficult for labelling 
        """
        if len(recs) > 0:
            arrows = make_arrows( recs[:,2,:3], recs[:,0,:3], mag=10 )
            pl.add_mesh( arrows, color="pink" )
            print(arrows)
        pass
    @classmethod
    def Draw1(cls, pl, recs): 
        """
        Old record layout 
        """
        rec_tloop_nodeIdx_ctrl = rec[3,:3].view(np.int32)
        arrows = make_arrows( rec_pos, rec_dir, mag=10 )
        pl.add_mesh( arrows, color="pink" )
        points = arrows.points 
        mask = slice(0,1)
        pl.add_point_labels( points[mask], [ str(rec_tloop_nodeIdx_ctrl) ] , point_size=20, font_size=36 )

    @classmethod
    def Draw(cls, pl, recs, ray_origin, ray_direction, off): 
        """
        :param pl: pyvista plotter
        :param recs: CSGRecord array for a single ray CSG tree intersection
        :param ray_origin:
        :param ray_direction:
        :param off: normalized vector out of the genstep plane, pointing to viewpoint  

        # less efficient but gives easy access to each arrow position 
        """

        pos_shifts = []
        for i in range(len(recs)): 
            left  = recs[i,0]
            n_left = left[:3]
            t_left  = np.abs(left[3])

            right = recs[i,1]
            n_right = right[:3]
            t_right = np.abs(right[3])

            tmin = recs[i,3,0] 
            t_min = recs[i,3,1] 
            tminAdvanced = recs[i-1,3,2] if i > 0 else 0. 
  
            ## tminAdvanced is bit confusing as collected 
            ## into record prior to the decision  (for leaf looping anyhow) 

            print("CSGRecord.Draw rec %d : tmin %10.4f t_min %10.4f tminAdvanced %10.4f (from prior rec) " % (i, tmin, t_min, tminAdvanced ))

            result = recs[i,4]
            t_result = np.abs(result[3])
            n_result = result[:3]

            pos_t_min = ray_origin + t_min*ray_direction                # base 
            pos_tmin  = ray_origin + tmin*ray_direction                 
            pos_tminAdvanced = ray_origin + tminAdvanced*ray_direction  
            pos_left  = ray_origin + t_left*ray_direction
            pos_right = ray_origin + t_right*ray_direction
            pos_result = ray_origin + t_result*ray_direction

            pos_shift_0 = float(i*10+0)*off 
            pos_shift_1 = float(i*10+1)*off 
            pos_shift_2 = float(i*10+2)*off 
            pos_shift_3 = float(i*10+3)*off 

            pos_shifts.append(pos_shift_0)

            if tminAdvanced > 0.:
                Rays.Draw(pl, pos_t_min, pos_tminAdvanced ,  ray_direction, ray_width=4, pos_shift=pos_shift_0, ray_color="green"  )
            pass

            Rays.Draw(pl, pos_tmin,  pos_left ,  n_left,        ray_width=4, pos_shift=pos_shift_1  )
            Rays.Draw(pl, pos_tmin,  pos_right,  n_right,       ray_width=4, pos_shift=pos_shift_2  )

            if t_result > 0.:
                Rays.Draw(pl, pos_tmin,  pos_result, n_result,      ray_width=4, pos_shift=pos_shift_3, ray_color="cyan", nrm_color="cyan"  )
            pass
        pass
        return pos_shifts
    pass   

    


pass 


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    pl = pv.Plotter(window_size=SIZE*2 ) 

    if 'EDL' in os.environ:
        print("EDL : enable_eye_dome_lighting :  improves depth perception with point clouds")
        pl.enable_eye_dome_lighting()  
    pass   

    plotbase = os.path.expandvars("$CFBASE/CSGIntersectSolidTest/$GEOM")
    print(" plotbase : %s " % plotbase )

    cegs_path = plotbase  
    #rel_name = "intersectSelected"
    rel_name = "saveCenterExtentGenstepIntersect"
    recs_path = os.path.join(plotbase, rel_name )



    cegs = SCenterExtentGenstep(cegs_path)
    fold = cegs.fold

    gspos = fold.gs[:,5,:3]

    gsid_ =  fold.gs[:,0,2].view(np.uint32).copy()
    gsid =  gsid_.view(np.int8).reshape(-1,4)   # q0.u.z (4 packed signed char) 

    no_gs = 'NO_GS' in os.environ 
    if no_gs:
        print(" skip gspos points due to NO_GS " )
    else:
        pl.add_points( gspos, color="yellow" )
    pass

    gsmeta = NPMeta(fold.gs_meta)  # TODO cxs_geochain uses genstep_meta : standardize
    gridspec = GridSpec(fold.peta, gsmeta)
    axes = gridspec.axes
    off = gridspec.off
    print(" gridspec.axes : %s   gridspec.off %s  " % (str(axes), str(off)) )

    zoom = efloat_("ZOOM", "1.")
    gridspec.pv_compose(pl, reset=False, zoom=zoom)

    no_isect = not hasattr(fold, 'isect') 
    gs_only = 'GS_ONLY' in os.environ
    quiet = True 

    if no_isect or gs_only:
       log.fatal("early exit just showing gensteps as no_isect:%s in fold OR gs_only:%d selected by GS_ONLY envvar" % (no_isect, gs_only))
       pl.show_grid()
       cp = pl.show()
       sys.exit(0)
    pass 

    dir_, t = fold.isect[:,0,:3], fold.isect[:,0,3]   # intersect normal and ray distance 
    pos, sd = fold.isect[:,1,:3], fold.isect[:,1,3]   # intersect position and surface distance 
    asd = np.abs(sd) 

    isect_gsid_ = fold.isect[:,3,3].copy().view(np.uint32)
    isect_gsid = isect_gsid_.view(np.int8).reshape(-1,4)    # genstep (ix,iy,iz,0) from which each intersect originated from     

    ## bitwise_and removes the bytes holding the photon index, for easier connection to gs level  
    isect_gsoid_ = np.bitwise_and( fold.isect[:,3,3].view(np.uint32), 0x00ffffff ).copy()
    isect_gsoid = isect_gsoid_.view(np.int8).reshape(-1,4)  
    np.all( isect_gsid[:,:3]  == isect_gsoid[:,:3] ) 

    ray_origin = fold.isect[:, 2, :3]
    ray_direction = fold.isect[:, 3, :3]

    asd_cut = 1e-3
    select_spurious = asd > asd_cut 
    select_all = t > 0.

    gss_ = np.unique(isect_gsoid_[select_spurious]) 
    gss = gss_.view(np.int8).reshape(-1,4)        # gensteps which have spurious intersects 
    gsid_idx = np.where(np.in1d(gsid_, gss_))[0]  # gsid_ indices of      
    gss_select = np.isin(gsid_, gss_)  # mask selecting  gensteps which have spurious intersects

    if no_gs:
        print(" skip gspos points due to NO_GS " )
    else:
        gss_select_count = np.count_nonzero(gss_select)
        if gss_select_count > 0:  
            pl.add_points( gspos[gss_select], color="green" )
        pass
        ## gs with spurious intersects presented differently
    pass

    select = select_all 

    count_all = np.count_nonzero(select_all)
    count_spurious = np.count_nonzero(select_spurious)

    cmdline = os.environ.get("CMDLINE", "no-CMDLINE")

    print("\n\n")
    print(cmdline)
    print("\n")
    print( "%40s : %d " % ("count_all", count_all) )
    print( "%40s : %d " % ("count_spurious", count_spurious) )

    sphi = 'SPHI' in os.environ
    if sphi:
        print(" SPHI : selecting intersects based on phi angle of the position ") 
        phi0,phi1 = efloatlist_("SPHI", "0.25,1.75")    
        cosPhi0,sinPhi0 = np.cos(phi0*np.pi),np.sin(phi0*np.pi)     
        cosPhi1,sinPhi1 = np.cos(phi1*np.pi),np.sin(phi1*np.pi)     
        PQ = cosPhi0*sinPhi1 - cosPhi1*sinPhi0 
        PR = cosPhi0*pos[:,1] - pos[:,0]*sinPhi0
        QR = cosPhi1*pos[:,1] - pos[:,0]*sinPhi1   #Q ^ R = cosPhi1*d.y - d.x*sinPhi1 
        select_phi = np.logical_and( PR > 0., QR < 0. ) if PQ > 0. else np.logical_or( PR > 0., QR < 0. )
        select_phi = np.logical_not( select_phi )
        count_phi = np.count_nonzero(select_phi)
        print( "%40s : %d " % ("count_phi", count_phi) )
        select = np.logical_and( select, select_phi )
        count_select = np.count_nonzero(select)
        print("%40s : %d   SPHI %s  phi0 %s phi1 %s PQ %s \n" % ("count_select", count_select, os.environ["SPHI"], phi0, phi1, PQ )) 
    pass

    ## when envvar IXYZ is defined restrict to a single genstep source of photons identified by grid coordinates
    ixyz = eintlist_("IXYZ", None)   
    sxyzw = eintlist_("SXYZW", None)   

    _sxyzw = np.array( sxyzw, dtype=np.int8 ).view(np.uint32)[0] if not sxyzw is None else None
    _sxyzw_idx = np.where( isect_gsid_ == _sxyzw )  if not _sxyzw is None else None


    if not ixyz is None:
        ix,iy,iz = ixyz
        select_isect_gsid = np.logical_and( np.logical_and( isect_gsid[:,0] == ix , isect_gsid[:,1] == iy ), isect_gsid[:,2] == iz )
        count_isect_gsid = np.count_nonzero(select_isect_gsid) 

        select = np.logical_and( select, select_isect_gsid )
        count_select = np.count_nonzero(select)

        print("%40s : %d   IXYZ %s  ix:%d iy:%d iz:%d" %   ("count_isect_gsid", count_isect_gsid, os.environ.get("IXYZ",None), ix,iy,iz ))
        print("%40s : %d  \n" % ("count_select", count_select)) 
    pass  

    ## when envvar IW is defined restrict to a single photon index, usually used together with IXYZ to pick one intersect
    iw = eint_("IW","-1")
    if iw > -1:
        # selecting a single intersect as well as single genstep  
        select_iw = isect_gsid[:,3] == iw 
        count_iw = np.count_nonzero(select_iw)
        select = np.logical_and( select, select_iw )
        count_select = np.count_nonzero(select)

        print("%40s : %d   IW %s " %   ("count_iw", count_iw, os.environ["IW"] ))
        print("%40s : %d  \n" % ("count_select", count_select)) 
    pass  

    # when envvar SPUR is defined restrict to intersects that have sd less than sd_cut  
    spurious = 'SPUR' in os.environ
    if spurious:
        select = np.logical_and( select, select_spurious )
        count_select = np.count_nonzero(select)
        print("%40s : %d    SPUR %s " % ("count_spurious", count_spurious, os.environ["SPUR"] ))
        print("%40s : %d  \n" % ("count_select", count_select)) 
    pass

    gsid_label = None 
    if not ixyz is None and not iw is None:
        ix,iy,iz = ixyz
        gsid_label = "gsid_%d_%d_%d_%d" % (ix,iy,iz,iw)     
        print(" IXYZ/IW defined =>  gsid_label : %s " % gsid_label )
    elif not sxyzw is None:
        sx,sy,sz,sw = sxyzw
        gsid_label = "gsid_%d_%d_%d_%d" % (sx,sy,sz,sw) 
        print(" SXYZW defined  =>  gsid_label : %s " % gsid_label )
    pass     
         
    if not gsid_label is None: 
        recs_fold = Fold.Load(recs_path, gsid_label, quiet=True)
        recs = [] if recs_fold is None else recs_fold.CSGRecord 
        if len(recs) > 0:
            print("%40s : %s   recs_fold.base %s " % ("recs", str(recs.shape), recs_fold.base )) 
        else:
            if quiet == False:
                print(" no recs to load at recs_path/gsid_label %s/%s " % (recs_path, gsid_label))
            pass
        pass
    else:
        recs = []
    pass


    pos_shifts = []
    if len(recs) > 0:
        assert not _sxyzw_idx is None
        _ray_origin = ray_origin[_sxyzw_idx][0]
        _ray_direction = ray_direction[_sxyzw_idx][0]
        pos_shifts = CSGRecord.Draw(pl, recs, _ray_origin, _ray_direction, off )
    pass

    count_select = np.count_nonzero(select)
    print("%40s : %d  \n" % ("count_select", count_select)) 

    s_t = t[select]
    s_pos = pos[select]
    s_sd = sd[select]
    s_isect_gsid = isect_gsid[select]
    s_dir = dir_[select]
    s_isect = fold.isect[select]
    s_ray_origin = ray_origin[select]
    s_ray_direction = ray_direction[select]
    s_pos_r = np.sqrt(np.sum(s_pos*s_pos, axis=1))  

    s_count = len(s_pos)
    s_limited = min( s_count, 100 )   
    # default number of photons per genstep is 100, so this gets all of those when selecting single genstep
    selected_isect = s_isect[:s_limited]

    print("%40s : %d  \n" % ("s_count", s_count)) 
    print("%40s : %d  \n" % ("s_limited", s_limited)) 
    print("%40s : %d  \n" % ("selected_isect", len(selected_isect))) 

    ## sub selection of the selected that are also spurious as judged by signed distance

    select_and_spurious = np.logical_and( select, select_spurious )
    ss_pos = pos[select_and_spurious]
    ss_dir = dir_[select_and_spurious]
    ss_ray_origin = ray_origin[select_and_spurious]

    ss_count = len(ss_pos)
    ss_limited = min( ss_count, 100 )   

    print("%40s : %d  \n" % ("ss_count", ss_count)) 
    print("%40s : %d  \n" % ("ss_limited", ss_limited)) 

    def fmt(i):
        _s_isect_gsid = "SXYZW=%d,%d,%d,%d  " % tuple(s_isect_gsid[i]) 
        _s_t = " s_t ( %10.4f ) " % s_t[i]
        _s_pos = " s_pos ( %10.4f %10.4f %10.4f ) " % tuple(s_pos[i])
        _s_pos_r = " s_pos_r ( %10.4f ) " % s_pos_r[i]
        _s_sd = " s_sd ( %10.4f ) " % s_sd[i]
        return " %-20s %s %s %s %s " % (_s_isect_gsid, _s_t, _s_pos, _s_pos_r, _s_sd )
    pass
    desc = "\n".join(map(fmt,np.arange(s_limited)))
    print(desc) 
    print(" Use IXYZ to select gensteps, IW to select photons within the genstep ")

    log.info( "asd_cut %10.4g sd.min %10.4g sd.max %10.4g num select %d " % (asd_cut, sd.min(), sd.max(), len(s_pos)))


    selected_isect_path = "/tmp/selected_isect.npy"
    key = 'SAVE_SELECTED_ISECT'
    save_selected_isect = key in os.environ
    if save_selected_isect and len(selected_isect) > 0 :
        print(" %s : saving selected_isect to %s " % (key, selected_isect_path)  )
        np.save( selected_isect_path, selected_isect )
    else:
        print(" define key %s to save selected isect to %s " % (key, selected_isect_path ))
    pass

    plot_selected = 'PLOT_SELECTED' in os.environ   # normally all positions are plotted not just selected
    if plot_selected:
        pl.add_points( s_pos, color="white" )
    else:
        if len(pos_shifts) == 0:
            pl.add_points( pos, color="white" )
        else:
            for pos_shift in pos_shifts:
                pl.add_points( pos+pos_shift, color="white" )
            pass
        pass
    pass

    ## different presentation of selected intersects and normals
    if s_count > 0 and sxyzw is None:
        Rays.Draw( pl, s_ray_origin[:s_limited], s_pos[:s_limited], s_dir[:s_limited] )
    pass
    if ss_count > 0 and sxyzw is None:
        Rays.Draw( pl, ss_ray_origin[:ss_limited], ss_pos[:ss_limited], ss_dir[:ss_limited], ray_color="yellow", nrm_color="yellow" )
    pass

    pl.show_grid()
    ReferenceGeometry(pl)
    topline = os.environ.get("TOPLINE", "topline")
    botline = os.environ.get("BOTLINE", "botline")
    pl.add_text(topline, position="upper_left")
    pl.add_text(botline, position="lower_left")

    figsdir = os.path.join(plotbase, "figs")
    if not os.path.isdir(figsdir):
        os.makedirs(figsdir)
    pass

    outname = "out.png" 
    outpath = os.path.join(figsdir, outname)
    print(" outpath : %s " % outpath )

    cp = pl.show(screenshot=outpath)


