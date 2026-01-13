#!/usr/bin/env python
"""
pvplt.py
===========

pvplt_simple
mpplt_focus
    given xlim, ymin args and FOCUS env-array return new xlim,ylim

pvplt_viewpoint
    according to EYE, LOOK, UP, ZOOM, PARA envvars
pvplt_photon
    pos, mom, pol plotting
pvplt_plotter
pvplt_arrows
pvplt_lines
pvplt_add_contiguous_line_segments

mpplt_add_contiguous_line_segments


pyvista plotter keys/mouse controls
------------------------------------

* https://docs.pyvista.org/api/plotting/plotting

* r : Reset the camera to view all datasets

* up/down : Zoom in and out

* mouse-wheel or right-click (mac: ctl+click) : Continuously zoom the rendering scene

  * BUT : when start too zoomed in this takes forever, instead use up/down
    for coarse zoom control and and then use mouse-wheel once have control

* shift+click or middle-click  (mac:shift+click): Pan the rendering scene

  * press shift and drag the mouse around to pan



* q : Close the rendering window
* f : Focus and zoom in on a point
* v : Isometric camera view
* w : Switch all datasets to a wireframe representation
* r : Reset the camera to view all datasets
* s : Switch all datasets to a surface representation
* shift+click or middle-click (mac:shift+click) : Pan the rendering scene
* left-click (mac:cmd+click) : Rotate the rendering scene in 3D
* ctrl+click : Rotate the rendering scene in 2D (view-plane)
* mouse-wheel or right-click (mac:ctl+click) : Continuously zoom the rendering scene
* shift+s : Save a screenshot (only on BackgroundPlotter)
* shift+c : Enable interactive cell selection/picking
* up/down : Zoom in and out
* +/- : Increase/decrease the point size and line widths


"""

import os, logging
log = logging.getLogger(__name__)
import numpy as np

## when have pyvista will almost certainly also have matplotlib but not conversely
## so defaulting to 2D MODE:2 is more sensible
D_ = os.environ.get("D", "2")
MODE_ = os.environ.get("MODE", D_ )
MODE = int(MODE_)
VERBOSE = int(os.environ.get("VERBOSE", "0")) == 1

print("pvplt MODE:%d " % (MODE))

mp = None
pv = None

if MODE in [2,3,-2,-3]:
    try:
        import matplotlib as mp
        import matplotlib.pyplot as mpp
        # this works in a varietry of versions
        # trying to control window position using  fig.canvas.manager.window
        # might need mp.use('TkAgg')  but that crashes
        #import matplotlib.pyplot as mp
    except ImportError:
        mp = None
        mpp = None
    pass
pass

if MODE in [3,-3]:
    try:
        import pyvista as pv
        #pv.global_theme.full_screen = True
    except ImportError:
        print("pvplt.py : FAILED to import pyvista MODE:%d" % MODE)
        pv = None
    pass
pass


if not mp is None:
    from matplotlib import collections  as mp_collections
    from matplotlib import lines as mp_lines
    from matplotlib.patches import Circle, Rectangle, Ellipse
else:
    mp_collections = None
    Circle = None
    Rectangle = None
    Ellipse = None
pass

from opticks.ana.eget import efloatlist_, efloatarray_, elookce_, elook_epsilon_, eint_
from opticks.ana.axes import Axes, X,Y,Z

themes = ["default", "dark", "paraview", "document" ]

if not pv is None:
    pv.set_plot_theme(themes[1])
pass

GUI = not "NOGUI" in os.environ

eary_ = lambda ekey, edef:np.array( list(map(float, os.environ.get(ekey,edef).split(","))) )
eintary_ = lambda ekey, edef:np.array( list(map(int, os.environ.get(ekey,edef).split(","))) )
efloat_ = lambda ekey, edef: float( os.environ.get(ekey,edef) )

SIZE = eintary_("SIZE", "1280,720")
ASPECT = float(SIZE[0])/float(SIZE[1])  # 1280/720 = 1.7777777777777777

XDIST = efloat_("XDIST", "200")
FOCUS = eary_("FOCUS", "0,0,0")
SCALE = efloat_("SCALE", "1")

def pvplt_simple(pl, xyz, label):
    """
    :param pl: pyvista plotter
    :param xyz: (n,3) shaped array of positions
    :param label: to place on plot
    """
    pl.add_text( "pvplt_simple %s " % label, position="upper_left")
    pl.add_points( xyz, color="white" )

def mpplt_focus_aspect(aspect=ASPECT, scale=SCALE):
    log.info("mpplt_focus_aspect aspect:%s FOCUS:%s scale:%s" % (str(aspect),str(FOCUS), scale))
    if np.all(FOCUS == 0): return None, None

    center = FOCUS[:2]
    extent = FOCUS[2] if len(FOCUS) > 2 else 100
    diagonal  = np.array([extent*aspect, extent])
    botleft = center - diagonal
    topright = center + diagonal
    log.info("mpplt_focus_aspect botleft:%s" % str(botleft))
    log.info("mpplt_focus_aspect topright:%s" % str(topright))
    xlim = np.array([botleft[0], topright[0]])*scale
    ylim = np.array([botleft[1], topright[1]])*scale
    return xlim, ylim



def mpplt_focus(xlim, ylim):
    """
    Used from::

        sysrap/sframe.py:mp_subplots (as used by tests/CSGSimtraceTest.py)
        g4cx/tests/cf_G4CXSimtraceTest.py:main

    :param xlim: array of shape (2,)
    :param ylim: array of shape (2,)
    :return xlim,ylim: restricted by FOCUS envvar
    """
    aspect = (xlim[1]-xlim[0])/(ylim[1]-ylim[0])
    log.info("mpplt_focus xlim:%s ylim:%s FOCUS:%s " % (str(xlim),str(ylim), str(FOCUS)))

    return xlim, ylim

def pvplt_viewpoint(pl, reset=False, verbose=False, m2w=None ):
    """


    https://github.com/pyvista/pyvista-support/issues/40

    """

    m2w = np.eye(4) if m2w is None else m2w
    assert m2w.shape == (4,4)

    _eye = np.ones(4)
    _eye[:3] = eary_("EYE",  "1,1,1")

    _look = np.ones(4)
    _look[:3] = eary_("LOOK",  "0,0,0")

    _up = np.zeros(4)
    _up[:3] = eary_("UP",  "0,0,1")

    zoom = efloat_("ZOOM", "1")

    eye  = np.dot( _eye,  m2w )
    look = np.dot( _look, m2w )
    up   = np.dot( _up,   m2w )


    PARA = "PARA" in os.environ
    if verbose:
        print("pvplt_viewpoint reset:%d PARA:%d " % (reset, PARA))
        print(" m2w\n%s " % str(m2w) )
        print(" _eye  : %s " % str(_eye) )
        print(" _look  : %s " % str(_look) )
        print(" _up  : %s " % str(_up) )
        print(" eye  : %s " % str(eye) )
        print(" look : %s " % str(look) )
        print(" up   : %s " % str(up) )
        print(" zoom : %s " % str(zoom) )
    pass
    if PARA:
        pl.camera.ParallelProjectionOn()
    pass
    pl.set_focus(    look[:3] )
    pl.set_viewup(   up[:3] )
    pl.set_position( eye[:3], reset=reset )
    pl.camera.Zoom(zoom)


def pvplt_photon( pl, p, polcol="blue", polscale=1, wscale=False, wcut=True  ):
    """
    :param p: array of shape (1,4,4)
    :param polcol:
    :param polscale:
    :param wscale: when True, using wavelength(Coeff) as scale for polarization vector
    :param wcut: when True, apply selection that wavelength(Coeff) must be greater than 0.
    """
    assert p.shape == (1,4,4)
    pos = p[:,0,:3]
    mom = p[:,1,:3]
    pol = p[:,2,:3]
    wav = p[:,2,3]

    if wcut and wav < 1e-6: return

    if wscale:
        polscale *= wav
    pass

    pl.add_points( pos, color="magenta", point_size=16.0 )

    lmom = np.zeros( (2, 3), dtype=np.float32 )
    lmom[0] = pos[0]
    lmom[1] = pos[0] + mom[0]

    lpol = np.zeros( (2, 3), dtype=np.float32 )
    lpol[0] = pos[0]
    lpol[1] = pos[0] + pol[0]*polscale

    pl.add_lines( lmom, color="red" )
    pl.add_lines( lpol, color=polcol )


def pvplt_plotter(label="pvplt_plotter", verbose=False, m2w=None):
    if verbose:
        print("STARTING PVPLT_PLOTTER ... THERE COULD BE A WINDOW WAITING FOR YOU TO CLOSE")
    pass

    WSIZE = SIZE*2
    print("pvplt_plotter WSIZE:%s" % repr(WSIZE))

    pl = pv.Plotter(window_size=WSIZE)
    pvplt_viewpoint(pl, reset=False, verbose=verbose, m2w=m2w)

    TEST = os.environ.get("TEST","")
    pl.add_text( "%s %s " % (label,TEST), position="upper_left")
    return pl

def pvplt_show(pl, incpoi=0., legend=False, title=None):
    """
    :param incpoi: float inrement to point size and line width, eg -5

    The below envvar take precedence over arguments::

        INCPOI=-5.
        LEGEND=1
        TITLE="The PyVista Window Title"


    This needs newer pyvista::

        x_pos_norm = 0.5
        y_pos_norm = 0.5
        text_scale = 20 # Adjust the text size to cover the area
        if LEGEND: pl.add_text(
               ' ' * 40, # Padding for width
                position=(x_pos_norm, y_pos_norm),
                font_size=text_scale,
                color='white',
                background='black',
                background_opacity=0.6,
                font_family='courier', # Monospace font helps keep the size consistent
                name='legend_background_box', # Give it a unique name
                viewport=True # Crucial: makes it relative to the screen, not 3D space
         )

    """
    TITLE = os.environ.get("TITLE", title)
    GRID = 1 == int(os.environ.get("GRID","0"))
    if GRID:
        bounds = efloatarray_("BOUNDS", "0,0,0,0,0,0" ) if "BOUNDS" in os.environ else None
        axes_ranges = efloatarray_("AXES_RANGES", "0,0,0,0,0,0" ) if "AXES_RANGES" in os.environ else None
        print("pvplt_show bounds [%s] " % str(bounds))
        print("pvplt_show axes_ranges [%s] " % str(axes_ranges))
        pl.show_grid(bounds=bounds, axes_ranges=axes_ranges )
    else:
        if VERBOSE: print("pvplt_show !(GRID==1) ")
    pass

    if "LINE" in os.environ:
        line = efloatarray_("LINE", "0,0,17000,0,0,22000")
        a = line[:3]
        b = line[3:]
        pvplt_add_line_a2b(pl, a, b)
    pass

    if "LINE2" in os.environ:
        line2 = efloatarray_("LINE2", "0,0,17000,0,0,22000")
        a = line2[:3]
        b = line2[3:]
        pvplt_add_line_a2b(pl, a, b)
    pass

    if "LINE3" in os.environ:
        lines = np.fromstring(os.environ["LINE3"],sep=",").reshape(-1,6)
        for line in lines:
            a = line[:3]
            b = line[3:]
            pvplt_add_line_a2b(pl, a, b)
        pass
    pass


    def callback_click_position(xy):
        print("callback_click_position xy : %s " % str(xy))
    pass
    pl.track_click_position(callback_click_position)


    POINT = np.fromstring(os.environ["POINT"],sep=",").reshape(-1,3) if "POINT" in os.environ else None
    POINTSIZE = float(os.environ.get("POINTSIZE",1.))

    BBOX = np.fromstring(os.environ["BBOX"],sep=",").reshape(-1,3) if "BBOX" in os.environ else None
    CIRCLE = np.fromstring(os.environ["CIRCLE"],sep=",").reshape(-1,4) if "CIRCLE" in os.environ else None
    NCIRCLE = np.fromstring(os.environ["NCIRCLE"],sep=",").reshape(-1,3) if "NCIRCLE" in os.environ else (0,1,0)

    if not POINT is None: pl.add_points(POINT, color="r", label="POINT", point_size=POINTSIZE) # point_size=POINTSIZE, render_points_as_spheres=True)
    if not BBOX is None: pl.add_points(BBOX, color="r", label="BBOX" )

    if not CIRCLE is None:
        for pcircle in CIRCLE:
            circle = pv.Disc(center=pcircle[:3], outer=pcircle[3], inner=0, normal=NCIRCLE, c_res=360)
            pl.add_mesh(circle, color='red', line_width=5, style='wireframe')
        pass
    pass

    INCPOI = float(os.environ.get("INCPOI",str(incpoi)))
    pl.increment_point_size_and_line_width(INCPOI)

    #pl.enable_point_picking(callback=pvplt_picked_point, use_picker=True)

    UL = os.environ.get("UL", "")
    if not UL is "": pl.add_text( UL, position="upper_left")
    LL = os.environ.get("LL", "")
    if not LL is "": pl.add_text( LL, position="lower_left")
    print("pvplt_show title:[%s] TITLE:[%s] UL:[%s] LL:[%s] " % (title, TITLE, UL, LL))


    LEGEND = bool(os.environ.get("LEGEND", legend))
    if LEGEND:
        vtk_actor = pl.add_legend(font_family="courier", size=(0.3,0.3))
        for i in range(vtk_actor.GetNumberOfEntries()):
            print(vtk_actor.GetEntryString(i))
        pass
    else:
        vtk_actor = None
    pass


    return pl.show(title=TITLE)

def pvplt_picked_point(picked_point, picker):
    print("pvplt_picked_point callback %s " % str(picked_point))
    point_idx = picker.GetPointId()
    print("pvplt_picked_point point_idx:%s " % point_idx)
pass


def mpplt_annotate_fig( fig, label  ):
    suptitle = os.environ.get("SUPTITLE",label)

    TOF = float(os.environ.get("TOF","0.99"))   # adjust the position of the title, to legibly display 4 lines

    if len(suptitle) > 0:
        log.debug("suptitle:%s " % (suptitle) )
        fig.suptitle(suptitle, family="monospace", va="top", ha='left', x=0.1, y=TOF, fontweight='bold' )
    else:
        log.debug("no SUPTITLE/label" )
    pass


def mpplt_annotate_ax( ax ):
    subtitle = os.environ.get("SUBTITLE", "")
    thirdline = os.environ.get("THIRDLINE", "")
    lhsanno  = os.environ.get("LHSANNO", "")
    rhsanno  = os.environ.get("RHSANNO", "")

    if len(subtitle) > 0:
        log.debug("subtitle:%s " % (subtitle) )
        ax.text( 1.05, -0.12, subtitle, va='bottom', ha='right', family="monospace", fontsize=12, transform=ax.transAxes)
    else:
        log.debug("no SUBTITLE")
    pass

    if len(thirdline) > 0:
        log.debug("thirdline:%s " % (thirdline) )
        ax.text(-0.05,  1.02, thirdline, va='bottom', ha='left', family="monospace", fontsize=12, transform=ax.transAxes)
    else:
        log.debug("no THIRDLINE")
    pass

    if len(lhsanno) > 0:
        log.debug("lhsanno:%s " % (lhsanno) )
        ax.text(-0.05,  0.01, lhsanno, va='bottom', ha='left', family="monospace", fontsize=12, transform=ax.transAxes)
    else:
        log.debug("no lhsanno")
    pass

    if len(rhsanno) > 0:
        rhsanno_pos = efloatarray_("RHSANNO_POS","0.6,0.01")
        log.debug("rhsanno:%s " % (rhsanno) )
        ax.text(rhsanno_pos[0], rhsanno_pos[1], rhsanno, va='bottom', ha='left', family="monospace", fontsize=12, transform=ax.transAxes)
    else:
        log.debug("no rhsanno")
    pass



def mpplt_plotter(nrows=1, ncols=1, label="", equal=True):
    """
    ISSUE: Observe that when plotting onto an external screen the layout of the
    plot differs : mysteriously this even happens when a plot is created on the
    laptop screen and then dragged onto the external monitor. But everything
    works fine when the screen capture is done on the laptop screen.
    This might be related to retina resolution of the laptop screen.


    The axs0 is placed into and np.array when (nrows, ncols) is (1,1)
    in order to make the return type uniform in all cases.
    Change type detection to check for np.ndarray as that
    is more stable than matplotlib type names.
    """

    if mpp is None:
        print("mplt_plotter : ERROR mpp is None : try again after conda hookup and activation\n")
    pass
    fig, axs0 = mpp.subplots(nrows=nrows, ncols=ncols, figsize=SIZE/100.) # 100 dpi

    if type(axs0) is np.ndarray:
        axs = axs0
    else:
        axs=np.array([axs0], dtype=object )
    pass

    if equal:
        for ax in axs:
            ax.set_aspect('equal')
        pass
    pass
    mpplt_annotate_fig(fig, label)
    for ax in axs:
        mpplt_annotate_ax(ax)
    pass
    return fig, axs


def plotter(label=""):
    if MODE == 2:
        pl = mpplt_plotter(label=label)
        assert type(pl) is tuple and len(pl) == 2  # fig, ax
    elif MODE == 3:
        pl = pvplt_plotter(label=label)
    else:
        pl = None
    pass
    return pl






def pvplt_arrows( pl, pos, vec, color='yellow', factor=0.15 ):
    """

    glyph.orient
        Use the active vectors array to orient the glyphs
    glyph.scale
        Use the active scalars to scale the glyphs
    glyph.factor
        Scale factor applied to sclaing array
    glyph.geom
        The geometry to use for the glyph

    """
    init_pl = pl == None
    if init_pl:
        pl = pvplt_plotter(label="pvplt_arrows")
    pass
    pos_cloud = pv.PolyData(pos)
    pos_cloud['vec'] = vec
    vec_arrows = pos_cloud.glyph(orient='vec', scale=False, factor=factor )

    pl.add_mesh(pos_cloud, render_points_as_spheres=True, show_scalar_bar=False)
    pl.add_mesh(vec_arrows, color=color, show_scalar_bar=False)


def pvplt_lines( pl, pos, vec, linecolor='white', pointcolor='white', factor=1.0 ):
    """
    :param pl: plotter
    :param pos: (n,3) array
    :param vec: (n,3) array

    2025/01 : changed args to use separate line and point color

    """
    init_pl = pl == None
    if init_pl:
        pl = pvplt_plotter(label="pvplt_line")
    pass
    pos_cloud = pv.PolyData(pos)
    pos_cloud['vec'] = vec
    geom = pv.Line(pointa=(0.0, 0., 0.), pointb=(1.0, 0., 0.),)
    vec_lines = pos_cloud.glyph(orient='vec', scale=False, factor=factor, geom=geom)
    pl.add_mesh(pos_cloud, color=pointcolor, render_points_as_spheres=True, show_scalar_bar=False)
    pl.add_mesh(vec_lines, color=linecolor, show_scalar_bar=False)


def pvplt_frame( pl, sfr, local=False ):
    extent = sfr.ce[3]
    m2w = sfr.m2w if local == False else np.eye(4)
    pvplt_frame_(pl, extent, m2w )

def pvplt_frame_(pl, e, m2w=np.eye(4) ):
    """

                 (-,+,+)
                   +-----------+ (+,+,+)
        (-,-,+)    |  (+,-,+)  |
            +----------2       |
            |      |   |       |
            |      |   |       |
            |      |   |       |
            |      +---|-------+ (+,+,-)      plus_y
            |          |
            0----------1       minus_y
       (-,-,-)      (+,-,-)

           -X         +X


            Z
            |   Y
            |  /
            | /
            |/
            O----- X

    """
    minus_y = np.array( [
           [-e,-e,-e],
           [+e,-e,-e],
           [+e,-e,+e],
           [-e,-e,+e],
           [-e,-e,-e] ])

    pvplt_add_contiguous_line_segments(pl, minus_y, m2w=m2w )

    plus_y = minus_y.copy()
    plus_y[:,1] = e

    pvplt_add_contiguous_line_segments(pl, plus_y, m2w=m2w )



def transform_points( pos3, transform ):
    assert len(pos3.shape) == 2 and pos3.shape[1] == 3 and pos3.shape[0] > 0
    opos = np.ones_like( pos3, shape=(len(pos3),4 ) )
    opos[:,:3] = pos3
    tpos = np.dot( opos, transform )
    return tpos[:,:3]


def pvplt_add_line_a2b(pl, a, b, color="white", width=1):
    lines = np.zeros( (2,3), dtype=np.float32 )
    lines[0] = a
    lines[1] = b
    pl.add_lines( lines, color=color, width=width  )




def pvplt_add_contiguous_line_segments( pl, xpos, point_size=1, point_color="white", line_color="white", m2w=None, render_points_as_spheres=False,  ):
    """
    :param pl: pyvista plotter
    :param xpos: (n,3) array of positions
    :param m2w: (4,4) transform array or None

    Adds red points at *xpos* and joins them with blue line segments using add_lines.
    This has been used only for small numbers of positions such as order less than 10
    photon step positions.
    """
    upos = xpos if m2w is None else transform_points( xpos, m2w )

    pl.add_points( upos, color=point_color, render_points_as_spheres=render_points_as_spheres, point_size=point_size )
    xseg = contiguous_line_segments(upos)
    pl.add_lines( xseg, color=line_color )


def mpplt_add_contiguous_line_segments(ax, xpos, axes, linewidths=2, colors="red", linestyle="dotted", label="label:mpplt_add_contiguous_line_segments", s=5 ):
    """
    :param ax: matplotlib 2D axis
    :param xpos: (n,3) array of positions
    :param axes: (2,)  2-tuple identifying axes to pick from the where X=0, Y=1, Z=2
    """

    if len(xpos) == 0:
        log.info("len(xpos) zero : skip ")
        return
    pass

    assert len(axes) == 2
    xseg = contiguous_line_segments(xpos[:,:3]).reshape(-1,2,3)  # (n,2,3)
    xseg2D = xseg[:,:,axes]   #  (n,2,2)    same as xseg[...,axes]  see https://numpy.org/doc/stable/user/basics.indexing.html

    lc = mp_collections.LineCollection(xseg2D, linewidths=linewidths, colors=colors, linestyle=linestyle )
    ax.add_collection(lc)

    xpos2d = xpos[:,axes]
    ax.scatter( xpos2d[:,0], xpos2d[:,1], s=s, label=label )


def mpplt_add_line(ax, a, b, axes ):
    """
    :param ax:  matplotlib 2D axis
    :param a: (3,) xyz position array
    :param b: (3,) xyz position array
    :param axes: (2,) 2-tuple identifying axes X=0, Y=1, Z=2
    """
    H,V = axes
    l = mp_lines.Line2D([a[H],b[H]], [a[V],b[V]])
    ax.add_line(l)


def mpplt_hist(mp, v, bins=100):
    fig, ax = mp.subplots()
    ax.hist(v, bins=bins )
    fig.show()


def get_ellipse_param(elpar):
    """
    :param elpar_: list of up to 7 float
    """
    elw = elpar[0]*2 if len(elpar) > 0 else 100
    elh = elpar[1]*2 if len(elpar) > 1 else 100
    elx = elpar[2]   if len(elpar) > 2 else 0
    ely = elpar[3]   if len(elpar) > 3 else 0
    ela = elpar[4]   if len(elpar) > 4 else 0.1
    ez0 = elpar[5]   if len(elpar) > 5 else 0.
    ez1 = elpar[6]   if len(elpar) > 6 else 0.
    return elw,elh,elx,ely,ela,ez0,ez1

def mpplt_add_ellipse_(ax, par, opt):
    """
    :param ax: matplotlib 2D axis
    :param elpar: list of up to 5 float

      tl            .
              .            .

      ml  +                      .

              .            .
      bl  +         .

    """
    elw,elh,elx,ely,ela,ez0,ez1 = get_ellipse_param(par)
    el = Ellipse(xy=[elx,ely], width=elw, height=elh, alpha=ela )
    ax.add_artist(el)

    #dz =  elh/2+ez0

    topleft = (-elw/2, elh/2 + ez0 )
    midleft = (-elw/2, ez0 )
    botleft = (-elw/2,-elh/2 + ez0 )
    ec = "red" if opt.find("red") > -1 else "none"

    log.info( " elw/2 %s elh/2 %s ez0 %s botleft %s ec %s opt %s" % (elw/2, elh/2, ez0, str(botleft), ec, opt) )

    if opt.find("top") > -1:
        clip_bx = Rectangle(midleft,elw,elh/2, facecolor="none", edgecolor=ec )
    elif opt.find("bot") > -1:
        clip_bx = Rectangle(botleft,elw,elh/2, facecolor="none", edgecolor=ec )
    else:
        clip_bx = None
    pass
    if not clip_bx is None:
        ax.add_artist(clip_bx)
        el.set_clip_path(clip_bx)
        ## clipping is selecting part of the ellipse to include (not to cut away)
    pass


def mpplt_add_ellipse(ax, ekey ):
    assert ekey.startswith("ELLIPSE")
    par = efloatlist_(ekey, "0,0,0,0")      #  elw,elh,elx,ely,ela,ez0,ez1
    opt = os.environ.get("%s_OPT" % ekey, "")

    if len(par) > 0 and par[0] > 0.:
        mpplt_add_ellipse_(ax, par, opt)
    pass



def get_rectangle_param(repar):
    """
    :param repar: list of up to 5 float
    """
    hx = repar[0] if len(repar) > 0 else 100
    hy = repar[1] if len(repar) > 1 else 100
    cx = repar[2] if len(repar) > 2 else 0
    cy = repar[3] if len(repar) > 3 else 0
    al = repar[4] if len(repar) > 4 else 0.2
    dy = repar[5] if len(repar) > 5 else 0
    return hx,hy,cx,cy,al,dy


def mpplt_add_rectangle_(ax, par, opt):
    """
                                   (cx+hx, cy+hy+dy)
           +----------+----------+
           |          |          |
           |          |(cx,cy+dy)|
           +----------+----------+
           |          |          |
           |          |          |
           +----------+----------+
       (cx-hx,cy-hy+dy)

    """
    hx,hy,cx,cy,al,dy = get_rectangle_param(par)
    botleft = (cx-hx, cy-hy+dy)
    width = hx*2
    height = hy*2
    bx = Rectangle( botleft, width, height, alpha=al, facecolor="none", edgecolor="red" )
    ax.add_artist(bx)


def mpplt_add_rectangle(ax, ekey):
    assert ekey.startswith("RECTANGLE")
    par = efloatlist_(ekey, "0,0,0,0")  #  hx,hy,cx,cy,al,dy
    opt = os.environ.get("%s_OPT" % ekey, "")
    mpplt_add_rectangle_(ax, par, opt)


def mpplt_add_shapes(ax):
    """
    """
    mpplt_add_ellipse(ax,"ELLIPSE0")
    mpplt_add_ellipse(ax,"ELLIPSE1")
    mpplt_add_ellipse(ax,"ELLIPSE2")

    mpplt_add_rectangle(ax,"RECTANGLE0")
    mpplt_add_rectangle(ax,"RECTANGLE1")
    mpplt_add_rectangle(ax,"RECTANGLE2")



def get_from_simtrace_isect( isect, mode="o2i"):
    """
    :param isect: (4,4) simtrace array item
    :param mode: str "o2i" "nrm" "nrm10"
    :return step: (2,3) point to point step
    """
    assert isect.shape == (4,4)

    dist = isect[0,3] # simtrace layout assumed, see CSG/tests/SimtraceRerunTest.cc
    nrm = isect[0,:3]
    pos = isect[1,:3]
    ori = isect[2,:3]
    mom = isect[3,:3]

    step = np.zeros( (2,3), dtype=np.float32 )
    if mode == "o2i":
        step[0] = ori
        step[1] = ori+dist*mom
    elif mode == "o2i_XDIST":
        step[0] = ori
        step[1] = ori+XDIST*mom
    elif mode == "nrm":
        step[0] = pos
        step[1] = pos+nrm
    elif mode == "nrm10":
        step[0] = pos
        step[1] = pos+10*nrm
    else:
        assert 0
    pass
    return step


def mpplt_simtrace_selection_line(ax, sts, axes, linewidths=2):
    """
    :param ax:
    :param sts: simtrace_selection array of shape (n,2,4,4)  where n is small eg < 10
    :param axes:

    The simtrace_selection created in CSG/tests/SimtraceRerunTest.cc
    contains pairs of isect the first from normal GPU simtrace and
    the second from CPU rerun.

    TODO: at dot at pos
    """
    pass
    log.info("mpplt_simtrace_selection_line sts\n%s\n" % repr(sts))

    MPPLT_SIMTRACE_SELECTION_LINE = os.environ.get("MPPLT_SIMTRACE_SELECTION_LINE", "o2i,o2i_XDIST,nrm10" )
    cfg = MPPLT_SIMTRACE_SELECTION_LINE.split(",")
    log.info("MPPLT_SIMTRACE_SELECTION_LINE %s cfg %s " % (MPPLT_SIMTRACE_SELECTION_LINE, str(cfg)))

    colors = ["red","blue"]
    if sts.ndim in (3,4) and sts.shape[-2:] == (4,4):
        for i in range(len(sts)):
            jj = list(range(sts.shape[1])) if sts.ndim == 4 else [-1,]
            log.info(" jj %s " % str(jj))
            for j in jj:
                isect = sts[i,j] if sts.ndim == 4 else sts[i]
                color = colors[j%len(colors)]
                if "o2i" in cfg:
                    o2i = get_from_simtrace_isect(isect, "o2i")
                    mpplt_add_contiguous_line_segments(ax, o2i, axes, linewidths=linewidths, colors=color, label="o2i", s=10 )
                pass
                if "o2i_XDIST" in cfg:
                    o2i_XDIST = get_from_simtrace_isect(isect, "o2i_XDIST")
                    mpplt_add_contiguous_line_segments(ax, o2i_XDIST, axes, linewidths=linewidths, colors=color, label="o2i_XDIST"  )
                pass
                if "nrm10" in cfg:
                    nrm10 = get_from_simtrace_isect(isect, "nrm10")
                    mpplt_add_contiguous_line_segments(ax, nrm10, axes, linewidths=linewidths, colors=color, label="nrm10" )
                pass
            pass
        pass

def pvplt_simtrace_selection_line(pl, sts):
    """
    """
    print("pvplt_simtrace_selection_line sts\n", sts)
    colors = ["red","blue"]
    if sts.ndim in (3,4) and sts.shape[-2:] == (4,4):
        for i in range(len(sts)):
            jj = list(range(sts.shape[1])) if sts.ndim == 4 else [-1,]
            for j in jj:
                isect = sts[i,j] if sts.ndim == 4 else sts[i]
                color = colors[j%len(colors)]
                o2i = get_from_simtrace_isect(isect, "o2i")
                nrm10 = get_from_simtrace_isect(isect, "nrm10")
                pvplt_add_contiguous_line_segments(pl, o2i, line_color=color, point_color=color )
                pvplt_add_contiguous_line_segments(pl, nrm10, line_color=color, point_color=color )
            pass
        pass
    pass

def ce_line_segments( ce, axes ):
    """
    :param ce: center-extent array of shape (4,)
    :param axes: tuple of length 2 picking axes from X=0 Y=1 Z=2
    :return box_lseg: box line segments of shape (4, 2, 3)


       tl            tr = c + [e,e]
         +----------+
         |          |
         |----c-----|
         |          |
         +----------+
       bl            br = c + [e,-e]

    """
    assert len(axes) == 2

    c = ce[:3]
    e = ce[3]

    h = e*Axes.UnitVector(axes[0])
    v = e*Axes.UnitVector(axes[1])

    tr = c + h + v
    bl = c - h - v
    tl = c - h + v
    br = c + h - v

    box_lseg = np.empty( (4,2,3), dtype=np.float32 )
    box_lseg[0] = (tl, tr)
    box_lseg[1] = (tr, br)
    box_lseg[2] = (br, bl)
    box_lseg[3] = (bl, tl)
    return box_lseg

def pvplt_parallel_lines(pl, gslim , aa, axes, look ):
    """
    HMM: seems no simple way to make dotted or dashed lines with pyvista ?
    The Chart system which has linestyles seems entirely separate from 3D plotting.

    For example with canonical XZ axes with envvar XX=x0,x1,x2
    will get 3 parallel vertical lines spanning between the gslim Z-limits
    at the provided x positions

         +---+------------+
         |   |            |
         |   |            |    Z
         |   |            |    |
         +---+------------+    +-- X
        x0  x1            x2


    gslim {0: array([209.35 , 210.197], dtype=float32), 1: array([-64.597, -64.597], dtype=float32), 2: array([129.514, 129.992], dtype=float32)}
    aa    {0: [209.5, 210.0], 1: [], 2: []}
    axes  (0, 2)
    look  [209.774, -64.59664, 129.752]
    """
    print("pvplt_parallel_lines")
    print("gslim %s " % str(gslim) )
    print("aa    %s " % str(aa) )
    print("axes  %s " % str(axes) )
    print("look  %s " % str(look) )
    assert len(axes) == 2
    H,V = axes
    for i in [X,Y,Z]:
        if len(aa[i]) == 0: continue
        for a in aa[i]:
            lo = look.copy()
            hi = look.copy()
            lo[i] = a
            hi[i] = a
            if i == H:  # i:horizontal axis, so create vertical lines
                lo[V] = gslim[V][0]
                hi[V] = gslim[V][1]
            elif i == V:  #  i:vertical axis, so create horizontal lines
                lo[H] = gslim[H][0]
                hi[H] = gslim[H][1]
            pass
            line = pv.Line(lo, hi, resolution=10)
            pl.add_mesh(line, color="w")
            log.info("i %d pv horizontal  lo %s hi %s " % (i, str(lo), str(hi)))
        pass
    pass



def mpplt_parallel_lines(ax, bbox, aa, axes, linestyle=None  ):
    """
    Draws axis parallel line segments in matplotlib and pyvista.
    The segments extend across the genstep grid limits.
    Lines to draw are configured using comma delimited value lists
    in envvars XX, YY, ZZ

    :param ax: matplotlib axis
    :param bbox: array of shape (3,2)
    :param aa: {X:XX, Y:YY, Z:ZZ } dict of values
    :param axes: tuple eg (0,2)

           +----------------------+
           |                      |
           |                      |
           +----------------------+
           |                      |
           +----------------------+
           |                      |
           |                      |   V=Z
           +----------------------+

             H=X

    """
    if ax is None: return
    H,V = axes
    log.info("mpplt_parallel_lines bbox[H] %s bbox[V] %s aa %s " % (str(bbox[H]), str(bbox[V]), str(aa)))
    for i in [X,Y,Z]:
        if len(aa[i]) == 0: continue
        for a in aa[i]:
            if V == i:  # vertical ordinate -> horizontal line
                ax.plot( bbox[H], [a,a], linestyle=linestyle )
            elif H == i:  # horizontal ordinate -> vertical line
                ax.plot( [a,a], bbox[V], linestyle=linestyle )
            pass
        pass
    pass

def mpplt_parallel_lines_auto(ax, bbox, axes, linestyle=None  ):
    aa = {}
    aa[X] = efloatlist_("XX")
    aa[Y] = efloatlist_("YY")
    aa[Z] = efloatlist_("ZZ")
    mpplt_parallel_lines(ax, bbox, aa, axes, linestyle=linestyle )




def mpplt_ce(ax, ce, axes, linewidths=2, colors="blue"):
    assert len(axes) == 2
    box_lseg = ce_line_segments(ce, axes)
    box_lseg_2D = box_lseg[:,:,axes]
    lc = mp_collections.LineCollection(box_lseg_2D, linewidths=linewidths, colors=colors, label="ce %10.4f" % ce[3])
    ax.add_collection(lc)

def pvplt_ce(pl, ce, axes, color="blue"):
    assert len(axes) == 2
    box_lseg = ce_line_segments(ce, axes)
    box_lseg_pv = box_lseg.reshape(-1,3)
    pl.add_lines( box_lseg_pv, color=color )


LOOKCE_COLORS = ["red","green", "blue", "cyan", "magenta", "yellow", "black"]
def lookce_color(i):
    return LOOKCE_COLORS[i%len(LOOKCE_COLORS)]

def mpplt_ce_multiple(ax, lookce, axes):
    for i, ce in enumerate(lookce):
        mpplt_ce(ax, ce, axes=axes,colors=lookce_color(i) )
    pass

def pvplt_ce_multiple(pl, lookce, axes):
    for i,ce in enumerate(lookce):
        pvplt_ce(pl, ce, axes=axes, color=lookce_color(i))
    pass



def contiguous_line_segments( pos ):
    """
    :param pos: (n,3) array of positions
    :return seg: ( 2*(n-1),3) array of line segments suitable for pl.add_lines

    Note that while the pos is assumed to represent a contiguous sequence of points
    such as photon step record positions the output line segments *seg*
    are all independent so they could be concatenated to draw line sequences
    for multiple photons with a single pl.add_lines

    For example, a set of three positions (3,3)::

        [0, 0, 0], [1, 0, 0], [1, 1, 0]

    Would give seg (2,2,3)::

         [[0,0,0],[1,0,0]],
         [[1,0,0],[1,1,0]]

    Which is reshaped to (4,3)::

         [[0, 0, 0], [1, 0, 0], [1, 0, 0], [1, 1, 0]])

    That is the form needed to represent line segments between the points
    with pl.add_lines

    """
    assert len(pos.shape) == 2 and pos.shape[1] == 3 and pos.shape[0] > 1
    num_seg = len(pos) - 1
    seg = np.zeros( (num_seg, 2, 3), dtype=pos.dtype )
    seg[:,0] = pos[:-1]   # first points of line segments skips the last position
    seg[:,1] = pos[1:]    # second points of line segments skipd the first position
    return seg.reshape(-1,3)


def test_pvplt_contiguous_line_segments():
    pos = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0]])
    seg = pvplt_contiguous_line_segments(pos)
    x_seg = np.array([[0, 0, 0], [1, 0, 0], [1, 0, 0], [1, 1, 0]])

    print(pos)
    print(seg)
    assert np.all( x_seg == seg )



def pvplt_check_transverse( mom, pol, assert_transverse=True ):
    mom_pol_transverse = np.abs(np.sum( mom*pol , axis=1 )).max()

    if mom_pol_transverse > 1e-5:
        print("WARNING mom and pol ARE NOT TRANSVERSE mom_pol_transverse %s assert_transverse %d " % ( mom_pol_transverse , assert_transverse ))
        if assert_transverse:
            assert mom_pol_transverse < 1e-5
        pass
    else:
        print("pvplt_check_transverse  mom_pol_transverse %s " % (mom_pol_transverse  ))
    pass




def pvplt_polarized( pl, pos, mom, pol, factor=0.15, assert_transverse=True ):
    """
    :param pl:
    :param pos: shape (n,3)
    :param mom: shape (n,3)
    :param pol: shape (n,3)
    :param factor: scale factor, needs experimentation for visibility


    https://docs.pyvista.org/examples/00-load/create-point-cloud.html
    https://docs.pyvista.org/examples/01-filter/glyphs.html

    Note bizarre issue of mom arrows only in one direction appearing ?

    Managed to get them to appear using add_arrows and fiddling with mag
    in CSG/tests/CSGFoundry_MakeCenterExtentGensteps_Test.py
    """
    pvplt_check_transverse(mom, pol, assert_transverse=assert_transverse)

    init_pl = pl == None
    if init_pl:
        pl = pvplt_plotter(label="pvplt_polarized")
    pass

    pos_cloud = pv.PolyData(pos)
    pos_cloud['mom'] = mom
    pos_cloud['pol'] = pol
    mom_arrows = pos_cloud.glyph(orient='mom', scale=False, factor=factor )
    pol_arrows = pos_cloud.glyph(orient='pol', scale=False, factor=factor )

    pl.add_mesh(pos_cloud, render_points_as_spheres=True, show_scalar_bar=False)
    pl.add_mesh(pol_arrows, color='lightblue', show_scalar_bar=False)
    pl.add_mesh(mom_arrows, color='red', show_scalar_bar=False)

    if init_pl:
        cp = pl.show() if GUI else None
    else:
        cp = None
    pass
    return cp

def pvplt_add_points_adaptive( pl, pos, **kwa ):
    p3 = None
    if pos.ndim == 2:
        p3 = pos[:,:3]
    elif pos.ndim == 3:
        p3 = pos[:,:,:3].reshape(-1,3)
    else:
        print("pvplt_add_points_adaptive UNEXPECTED pos.ndim : %d " % pos.ndim )
    pass
    return pvplt_add_points(pl, p3, **kwa )


def pvplt_add_points( pl, pos, **kwa ):
    if pos is None or len(pos) == 0:
        if VERBOSE: print("pvplt_add_points len(pos) ZERO   %s " % repr(kwa))
        return None
    pass
    #print("pvplt_add_points pos.shape %s kwa %s " % ( str(pos.shape), str(kwa) ))
    pos_cloud = pv.PolyData(pos)
    pl.add_mesh(pos_cloud, **kwa )
    return pos_cloud


def pvplt_make_delta_lines( pos, delta ):
    """
    :param pos: (n,3) coordinates
    :param delta: (n,3) offsets
    :return vlin, lines:

    vlin
        (n,3) vertices of start and end points of the lines

    lines
        specification with number of points to join (2)
        and indices into vlin, eg::

             np.array( [[2,100,101], [2,102,103]] ).ravel()

    """
    assert(len(pos) == len(delta))

    num_line = len(pos)
    vlin = np.zeros([num_line,2,3])
    vlin[:,0] = pos
    vlin[:,1] = pos + delta
    vlin = vlin.reshape(-1,3)

    _lines = np.zeros( (num_line,3), dtype=np.int64 )
    _lines[:,0] = 2                           # 2 indices for each line segment
    _lines[:,1] = np.arange(0, 2*num_line,2)  # line start index within vlin
    _lines[:,2] = 1 + _lines[:,1]               # line end index within vlin
    lines = _lines.ravel()

    return vlin, lines


def pvplt_add_delta_lines( pl, pos, delta, **kwa ):
    """
    :param pl: plotter
    :param pos: (n,3) coordinates
    :param delta: (n,3) offsets
    """
    vlin, lines = pvplt_make_delta_lines(pos, delta )
    pvplt_add_lines(pl, vlin, lines, **kwa )


def pvplt_add_lines( pl, pos, lines, **kwa ):
    """
    :param pos: float array of shape (n,3)
    :param lines: int array of form np.array([[2,0,1],[2,1,2]]).ravel()

    Used for adding intersect normals
    """
    if len(pos) == 0:
        if VERBOSE: print("pvplt_add_lines len(pos) ZERO   %s " % repr(kwa))
        return None
    pass
    if VERBOSE: print("pvplt_add_lines pos.shape %s lines.shape %s kwa %s " % ( str(pos.shape), str(lines.shape), str(kwa) ))
    pos_lines = pv.PolyData(pos, lines=lines)
    pl.add_mesh(pos_lines, **kwa )
    return pos_lines

if __name__ == '__main__':
    test_pvplt_contiguous_line_segments()
pass

