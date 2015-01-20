#!/bin/env python
"""
Shortcut functions for use from ipython. Usage::

    In [2]: from env.g4dae import chroma_geometry, geometry, daenode

    In [3]: g = geometry()

    In [2]: dae = daenode()



    In [12]: slow = l.extra.properties['SLOWCOMPONENT']

    In [13]: fast = l.extra.properties['FASTCOMPONENT']

    In [21]: plt.bar(slow[:,0], slow[:,1])
    Out[21]: <Container object of 275 artists>

    In [22]: plt.show()




"""
import os, logging
log = logging.getLogger(__name__)
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from env.geant4.geometry.collada.g4daeview.daedirectconfig import DAEDirectConfig
from env.geant4.geometry.collada.g4daenode import DAENode
from env.geant4.geometry.collada.g4daeview.daegeometry import DAEGeometry
from chroma.detector import Detector

from env.g4dae.types import NPY, pro_, Prop
from env.g4dae.types import CerenkovStep, G4CerenkovPhoton, ChCerenkovPhoton
from env.g4dae.types import ScintillationStep, G4ScintillationPhoton, ChScintillationPhoton

a4inches = np.array((11.69, 8.28)) 

cg = None
def cg_get():
    global cg
    if cg is None:
        cg = chroma_geometry()
    return cg 

g = None
def g_get():
    global g
    if g is None:
        g = geometry()
    return g

dae = None
def dae_get():
    global dae
    dae = daenode()
    return dae

def config():
    cfg = DAEDirectConfig()
    cfg.parse(nocli=True)
    return cfg

def daenode():
    cfg = config()
    DAENode.init(cfg.path)
    return DAENode

def chroma_geometry():
    cfg = config()
    print cfg.chromacachepath
    cg = Detector.get(cfg.chromacachepath)
    return cg 

def geometry():
    cfg = config()
    g = DAEGeometry.get(cfg) 
    return g 

def get_gdls():
    dae = dae_get()
    return dae.materialsearch("__dd__Materials__GdDopedLS")    

def get_ls():
    dae = dae_get()
    return dae.materialsearch("__dd__Materials__LiquidScintillator")
 
def plt_gdls():
    gdls = get_gdls()
    props = gdls.extra.properties
    fast = props['FASTCOMPONENT']
    slow = props['SLOWCOMPONENT']

    plt.title( "GdLS  ln(SLOWCOMPONENT) vs wl   ")
    plt.plot(slow[:,0],np.log(slow[:,1]), 'r+')
    plt.show()

 



def genconsistency(evt):
    genconsistency_cerenkov(evt)
    genconsistency_scintillation(evt)    


def genconsistency_cerenkov(evt):
    g4c = G4CerenkovPhoton.get(evt)
    chc = ChCerenkovPhoton.get(evt)
    stc = CerenkovStep.get(evt)
    n = stc.totPhotons

    log.info("g4c : GOPCERENKOV    %s " % str(g4c.shape))
    log.info("chc :  OPCERENKOV    %s " % str(chc.shape))
    log.info("stc :    CERENKOV    %s ==> N %s " % (str(stc.shape), n) )
    
    assert g4c.shape[0] == n
    assert chc.shape[0] == n


def genconsistency_scintillation(evt):
    g4s = G4ScintillationPhoton.get(evt)
    chs = ChScintillationPhoton.get(evt)
    sts = ScintillationStep.get(evt)
    n = sts.totPhotons

    log.info("g4s : GOPSCINTILLATION    %s " % str(g4s.shape))
    log.info("chs :  OPSCINTILLATION    %s " % str(chs.shape))
    log.info("sts :    SCINTILLATION    %s ==> N %s " % (str(sts.shape), n) )
    
    assert g4s.shape[0] == n
    assert chs.shape[0] == n








def g4_cerenkov_wavelength(tag, **kwa):
    """
    """
    cg = cg_get()
    pass
    g4c = G4CerenkovPhoton.get(tag)
    base = os.path.expandvars('$STATIC_BASE/env/g4dae') 
    path =  os.path.join(base, "g4_cerenkov_wavelength.png")
    cat = "cmat"
    val = "wavelength"
    title = "G4/Detsim Generated Cerenkov Wavelength by material" 

    catplot(g4c, cat=cat, val=val, path=path, title=title, log=True, histtype='step', stacked=False)



def generated_scintillation_3xyzw(tag=1):
    typs = "gopscintillation opscintillationgen"
    suptitle = "GPU Generated Scintillation Photons(blue) Compared to Geant4(red) (Single Event)"
    cf('3xyzw', tag=tag, typs=typs, legend=False, log=True, suptitle=suptitle )

def generated_cerenkov_3xyzw(tag=1):
    typs = "gopcerenkov opcerenkovgen"
    suptitle = "GPU Generated Cerenkov Photons(blue) Compared to Geant4(red) (Single Event)"
    cf('3xyzw', tag=tag, typs=typs, legend=False, log=True, suptitle=suptitle )


def generated_scintillation_time_wavelength(tag=1):
    typs = "gopscintillation opscintillationgen"
    suptitle = "GPU Generated Scintillation Photons Compared to Geant4 (Single Event)"
    path = "generated_scintillation_time_wavelength"
    cf('time_wavelength',tag=tag, typs=typs, legend=[True, False], log=True, suptitle=suptitle, path=path)

def generated_cerenkov_time_wavelength(tag=1):
    typs = "gopcerenkov opcerenkovgen"
    suptitle = "GPU Generated Cerenkov Photons Compared to Geant4 (Single Event)"
    path = "generated_cerenkov_time_wavelength"
    cf('time_wavelength',tag=tag, typs=typs, legend=[True, False], log=True, suptitle=suptitle, path=path)






pdgcode = {11:"e",13:"mu",22:"gamma"}
scntcode = {1:"fast",2:"slow"}
def catname(cat,ic):
    if cat == 'cmat':
        cg = cg_get()
        material = cg.unique_materials[ic]
        name = material.name[17:-9]
    elif cat == 'pdg':
        name = pdgcode.get(ic, ic) 
    elif cat == 'scnt':
        
        name = scntcode.get(ic, ic) 
    else:
        name = "%s:%s" % (cat,ic)
    pass
    return name 



def plt_save(path):
    if path is None:return
    base = os.path.expandvars('$STATIC_BASE/env/g4dae') 
    path =  os.path.join(base, "%s.png" % path)

    log.info("saving to %s " % path)
    dirp = os.path.dirname(path)
    if not os.path.exists(dirp):
        os.makedirs(dirp)
    pass 
    plt.savefig(path)


def plt_figure(cfg):
    figsize = cfg.pop('figsize', None)
    if figsize is None:
        fig = plt.figure()
    else:
        fig = plt.figure(figsize=figsize)    
    pass
    return fig 



def catplot(a, **kwa):
    """
    Category plot, eg Geant4 Generated Cerenkov Wavelength categorized by material  

    ::

         g4s, = NPY.mget(1,"gopscintillation")
         catplot(g4s, val='wavelength', cat='pdg' )

    """
    cfg = dict(bins=100,cat='aux0',val='wavelength',ics=None, reverse=True, path=None, title=None)
    cfg['figsize'] = a4inches*0.8
    cfg.update(kwa)

    plt_figure(cfg)    

    title = cfg.pop('title')
    if title is None:
        title = " %s (%s categories)" % (cfg['val'], cfg['cat'])

    plt.title(title)

    cat = cfg.pop('cat')
    val = cfg.pop('val')
    ics = cfg.pop('ics')
    reverse = cfg.pop('reverse')
    path = cfg.pop('path')

    catprop = getattr(a, cat)
    valprop = getattr(a, val)
    if ics is None:
        ics = np.unique(catprop)
    else:
        log.info("using argument ics")
    pass
    bc = np.bincount(catprop)

    print "ics:", ics
    cfg['label'] = "All [%d]" % bc.sum()
    plt.hist(valprop, **cfg)
    for ic in sorted(ics, key=lambda ic:bc[ic], reverse=reverse):
        cfg['label'] = "%20s  [%d]" % (catname(cat,ic),bc[ic])
        print cfg
        plt.hist(valprop[catprop == ic], **cfg)
    pass
    plt.legend()

    plt_save(path)

    plt.show()


def cf_cerenkov(qty='wavelength', tag=1, **kwa):
    """
    ::

       cf_cerenkov('wavelength')
       cf_cerenkov('time')

    """
    g4c,chc = NPY.mget(tag, "gopcerenkov","opcerenkov")
    cf(qty, g4c, chc, **kwa)

def cf_scintillation(qty='wavelength', tag=1, **kwa):
    """
    ::

       cf_scintillation('wavelength')
       cf_scintillation('time')

    """
    g4s = G4ScintillationPhoton.get(tag)
    chs = ChScintillationPhoton.get(tag)
    cf(qty, g4s, chs, **kwa)




def plot_refractive_index(tag=1, **kwa):
    """
    G4/Detsim
       Scintillators start at 80nm, waters at 200nm

    Chroma Standard interpolated 
        Everything interpolated to start from 60nm

    """
    cs = CerenkovStep.get(tag)
    cg = cg_get()
    mm = cs.materials(cg)
    cfg = dict(qty='refractive_index')
    cfg.update(kwa)
    qplot(mm, **cfg)


def plot_refractive_index_comparison(tag=1, **kwa):
    cs = CerenkovStep.get(tag)
    cg = cg_get()
    mm = cs.materials(cg)
    cfg = dict(qty='refractive_index')
    cfg.update(kwa)

    nr, nc = 1, 2 

    plt.subplot(nr, nc, 1 ) 
    qplot(mm, **cfg)

    plt.subplot(nr, nc, 2 ) 
    cfg.update(standard=True)
    qplot(mm, **cfg)




def qplot(materials, standard=False, qty='refractive_index'):
    """
    :param materials: list of chroma material instances
    :param standard:  when True apply chroma wavelength standardization and interpolation
    :param qty: name of quantity 
    """
    title = qty
    if standard:
        title += " standardized " 

    for m in materials:
        q = getattr(m, qty, None)
        if q is None:continue
        if standard:
            q = standardize(q)
        pass
        plt.plot( q[:,0], q[:,1], label=m.name[17:-9])
        pass
    pass
    plt.title(title)
    plt.legend()


def water_indices(cg):
    return filter(lambda _:cg.unique_materials[_].name.find('Water')>-1,range(len(cg.unique_materials)))

def chroma_refractive_index():
    for im, cm in enumerate(cg.unique_materials):
        wlri = cm.refractive_index
        wl = wlri[:,0]
        ri = wlri[:,1]
        print "[%2d] %25s %10s     wl %7.2f : %7.2f     %10.3f : %10.3f " % ( im, cm.name[17:-9], str(wlri.shape), wl.min(), wl.max(), ri.min(), ri.max() )  


def cerenkov_wavelength(cs, csi=0, nrand=100000, standard=False):
    """
    ::

         cg = chroma_geometry()
         cs = stc(1)

    Rapidly descending distrib with wavelength (Cerenkov blue light)
    starting from the low edge of the ri property of the material.
    What you get is majorly dependent on the ri range of the material
    so if diffent materials have different ranges, artifacts are inevitable

    Scintillator RINDEX start at 80nm, waters at 200nm

    ::

        In [56]: cerenkov_wavelength(cg, cs, 0)
        materialIndex 24 BetaInverse 1.00001 maxSin2 0.482422 material __dd__Materials__IwsWater0xc288f98 
        w0 199.975 w1 799.898 

        In [57]: cerenkov_wavelength(cg, cs, 1)
        materialIndex 24 BetaInverse 1.00001 maxSin2 0.482422 material __dd__Materials__IwsWater0xc288f98 
        w0 199.975 w1 799.898 

        In [58]: cerenkov_wavelength(cg, cs, 1000)
        materialIndex 0 BetaInverse 1.41302 maxSin2 0.0550548 material __dd__Materials__LiquidScintillator0xc2308d0 
        w0 79.9898 w1 799.898 


    """
    materialIndex = cs.materialIndex[csi]
    BetaInverse = cs.BetaInverse[csi]
    maxSin2 = cs.maxSin2[csi]

    material = cg.unique_materials[materialIndex]
    ri = material.refractive_index

    if standard:
        ri = standardize(ri) 

    w0 = ri[0,0]
    w1 = ri[-1,0]

    print "materialIndex %s BetaInverse %s maxSin2 %s material %s " % (materialIndex, BetaInverse, maxSin2, material.name)
    print "w0 %s w1 %s " % (w0, w1) 

    u1 = np.random.random(nrand)
    u2 = np.random.random(nrand)
 
    iw = (1./w1)*u1 + (1./w0)*(1.-u1)  # uniform in 1/w 
    w = 1./iw

    sampledRI = np.interp( w, ri[:,0], ri[:,1] )
    cosTheta = BetaInverse/sampledRI
    sin2Theta = (1.0 - cosTheta)*(1.0 + cosTheta)
    sin2Theta_over_maxSin2 = sin2Theta/maxSin2 
    ws = w[np.where( u2 <= sin2Theta_over_maxSin2)]

    plt.hist(ws, bins=100)
    plt.show() 





def get_cheat_cdf(name="gdls_fast"):
    """
    *  (1/wavelength, ascending_cdf)

    ::

        In [20]: ccdf = get_cheat_cdf()
    
        In [24]: ccdf
        Out[24]: 
        array([[ 0.001,  0.   ],
               [ 0.002,  0.001],
               [ 0.002,  0.001],
               ...
               [ 0.005,  1.   ],
               [ 0.008,  1.   ],
               [ 0.013,  1.   ]], dtype=float32)

        In [23]: plt.plot(ccdf[:,0],ccdf[:,1])

    """
    cdf = pro_(name).copy()      

    cdf[:,1] = cdf[:,1]/cdf[:,1].max()   # make y range 0:1

    return cdf


def scintillation_wavelength(n=2817543):
    """
    The "cheats" by using the purloined ScintillationIntegral 
    (ie does not form the cumulative summation from the input property
    itself)

    So just need to sample from the CDF,  
    a uniform draw in "y" is used to lookup the "x"  
    (1/wavelength)

    Where the reciprocal is taken doesnt matter for middle and high wavelengths, 
    but has significant effect on low wavelengths 200:350 nm
    reciprocal after (ie the interpolation yields reciprocal wavelengths)
    being an almost pefect match for the G4 distrib.

    Thus to get a match need to do interpolation in 1/wavelength

    ::

       plt.figure()
       plt.title("reciprocal after")
       plt.hist(wa_(n), bins=100, log=True, range=(100,900), histtype="step")  
       plt.hist(wa_(n), bins=100, log=True, range=(100,900), histtype="step")  
       plt.hist(wa_(n), bins=100, log=True, range=(100,900), histtype="step")  

       plt.figure()
       plt.title("reciprocal before")
       plt.hist(wb_(n), bins=100, log=True, range=(100,900), histtype="step")  
       plt.hist(wb_(n), bins=100, log=True, range=(100,900), histtype="step")  
       plt.hist(wb_(n), bins=100, log=True, range=(100,900), histtype="step")  
       ...

       g4s = G4ScintillationPhoton.get(1)
       wg = g4s.wavelength

       plt.figure(1)
       plt.hist( wg, bins=100, log=True, range=(100,900), histtype="step")

       plt.figure(2)
       plt.hist( wg, bins=100, log=True, range=(100,900), histtype="step")


    """
    ccdf = get_cheat_cdf("gdls_fast")   # (1/wavelength, ascending_cdf)

    wa_ = lambda n:1/np.interp( np.random.rand(n) , ccdf[:,1], ccdf[:,0] ) # reciprocal after 

    wb_ = lambda n:np.interp( np.random.rand(n) , ccdf[:,1], 1/ccdf[:,0] ) # reciprocal before 

    plt.hist(wa_(n), bins=100, log=True, range=(100,900), histtype="step")

    plt.show()



def construct_cdf_energywise(xy):
    """
    Duplicates DsChromaG4Scintillation::BuildThePhysicsTable     

    # NB changed to return (asc 1/wavelenth, asc cdf) 
 
    """
    assert len(xy.shape) == 2 and xy.shape[-1] == 2

    bcdf = np.empty( xy.shape )

    rxy = xy[::-1]              # reverse order, for ascending energy 

    x = 1/rxy[:,0]              # work in inverse wavelength 1/nm

    y = rxy[:,1]

    ymid = (y[:-1]+y[1:])/2     # looses entry as needs pair

    xdif = np.diff(x)            

    #bcdf[:,0] = rxy[:,0]        # back to wavelength
    bcdf[:,0] = x                # keeping 1/wavelenth

    bcdf[0,1] = 0.

    np.cumsum(ymid*xdif, out=bcdf[1:,1])

    bcdf[1:,1] = bcdf[1:,1]/bcdf[1:,1].max() 

    return bcdf    # (asc 1/wavelength, asc cdf)



def get_cdf():
    """
    Original and reversed plot look precisely the same, but the 
    reversed enables interp to work.

         plt.plot(cdf[:,0],cdf[:,1])
         plt.plot(cdf[::-1,0],cdf[::-1,1])

    """
    #from env.geant4.geometry.collada.collada_to_chroma import construct_cdf_energywise

    ls = get_ls()

    fast = ls.extra.properties['FASTCOMPONENT'].astype(np.float64)     

    cdf = construct_cdf_energywise(fast)  # (asc 1/wavelength, asc cdf )
   
    return cdf



def compare_cdf():
    ccdf = get_cheat_cdf()         # (asc 1/wavelength, asc cdf)

    rcdf = get_cdf()               # (asc 1/wavelength, asc cdf) 

    assert np.allclose(rcdf,ccdf)  #  max diff ~1.e-7
    


def scintillation_wavelength_raw(n=2817543, plot=True, standard=False, wl="60:810:20", after=True):
    """
    Succeeds to reproduce the scintillation wavelength distrib 
    by duplicating the DsChromaG4Scintillation::BuildThePhysicsTable 
    logic in construct_cdf_energywise

    ::

        plt.ion()
        scintillation_wavelength_raw()
        scintillation_wavelength_raw(after=False)                 # very off 200:350, 700:800
        scintillation_wavelength_raw(after=False, standard=True)  # huh standard mode mends the "before"

        scintillation_wavelength_raw(standard=True,wl="60:810:20")   # slightly off at ~400nm cliff
        scintillation_wavelength_raw(standard=True,wl="60:810:10")   # step of 10nm much better 

        ## good agreement 

    """
    cdf = get_cdf()

    if standard:
        cdf = standardize(cdf, wl, reciprocal=True)

    wb_ = lambda n:np.interp( np.random.rand(n) , cdf[:,1], 1/cdf[:,0] ) # reciprocal before 
    wa_ = lambda n:1/np.interp( np.random.rand(n) , cdf[:,1], cdf[:,0] ) # reciprocal after 

    ## huh after/before seems making no difference here
    if after:
        w_ = wa_
    else:
        w_ = wb_

    w = w_(n)   

    if plot:
        plt.hist(w, bins=100, log=True, range=(100,900), histtype='step')

    return w


def interp_material_property(domain, prop, reciprocal=False):
    """
    :param domain:
    :param prop:

    from chroma.gpu.GPUGeometry
    note that it is essential that the material properties be
    interpolated linearly. this fact is used in the propagation
    code to guarantee that probabilities still sum to one.
    """
    ascending_ = lambda _:np.all(np.diff(_) >= 0)
    assert ascending_(domain) 
    assert ascending_(prop[:,0]) 
    return np.interp(domain, prop[:,0], prop[:,1]).astype(np.float32)



def get_wavelengths(wl="60:810:20"):
    wavelengths = np.arange(*map(float,wl.split(":"))).astype(np.float32)
    return wavelengths


def standardize( prop, wl="60:810:20", reciprocal=False):
    """
    :param reciprocal:  when true the interpolation assumes prop[:,0] to be in 1/wavelength

    mimic what the chroma.geometry machinery does to properties on copying to GPU
    """
    wavelengths = get_wavelengths(wl)

    domain = 1/wavelengths[::-1] if reciprocal else wavelengths

    vals = interp_material_property(domain,  prop, reciprocal=reciprocal)

    return np.vstack([domain, vals]).T


def demo_standardize():
    """
    Stepsize of 20 does poor job at ~400nm
    """
    rcdf = get_cdf()

    s_rcdf = standardize(rcdf, reciprocal=True, wl="60:810:10")

    plt.plot(1/rcdf[:,0],rcdf[:,1],'r-+',1/s_rcdf[:,0],s_rcdf[:,1], 'b-+' )


def sample_reciprocal_cdf( u, nbin, x0, delta, cdf_y ):
    """
    Due to the 1 to 1 mirror bin relationship 
    between the domain and its reciprocal, can 
    use a CDF on the reciprocal domain via reading 
    from the right.

    ::

        In [52]: x = np.arange(1,5,1).astype(np.float32)

        In [53]: 1/x[::-1]
        Out[53]: array([ 0.25 ,  0.333,  0.5  ,  1.   ], dtype=float32)

        In [54]: x
        Out[54]: array([ 1.,  2.,  3.,  4.], dtype=float32)

    """

    lower, upper = 0, nbin - 1

    while lower < upper-1:
        half = (lower + upper)//2
        y = cdf_y[half]
        if u < y:
            upper = half
        else:
            lower = half
        pass
    pass

    #found the bin in which the draw lies
    delta_cdf_y = cdf_y[upper] - cdf_y[lower]

    #domain is 1/wavelength[::-1] so looking from the right 
    # within the bin, upwards fraction   
      
    r_fraction = (cdf_y[upper]-u)/delta_cdf_y    
    r_upper = nbin - 1 - upper ;   
 
    return x0 + delta*r_upper + delta*r_fraction ;


def test_sample_reciprocal_cdf(wl="60:810:10"):

     cdf = get_cdf()

     reciprocal = True

     wavelengths = get_wavelengths(wl)

     domain = 1/wavelengths[::-1] if reciprocal else wavelengths

     scdf = standardize(cdf, reciprocal=reciprocal, wl=wl)

     nbin = len(wavelengths)
     x0 = wavelengths[0]
     delta = np.unique(np.diff(wavelengths)).item()

     sample_reciprocal_cdf( 0.5 , nbin, x0, delta, scdf[:,1] ) 




def cf(qty, *arys, **kwa):
    """
    Comparison histogram with legend  

    ::

        cf('wavelength', tag=1, typs="test gopcerenkov", legend=True)

        cf('3xyz', tag=1, typs="gopcerenkov opcerenkov test", legend=False)

        cf('3xyzw', tag=1, typs="opcerenkov gopcerenkov test", legend=False)

        cf('wavelength', g4s, chs, log=True)


    """
    if len(arys) == 0:
        tag  = kwa.pop('tag')
        typs = kwa.pop('typs')
        arys = NPY.mget(tag, typs) 
    pass

    cfg = dict(bins=100,histtype="step",title=qty, color="rbgcmyk", legend=True)
    cfg.update(kwa)
    cfg['figsize'] = a4inches*0.8

    if qty == '3xyz':
        qty = "posx posy posz dirx diry dirz polx poly polz"  
    elif qty == '3xyzw':
        qty = "posx posy posz time dirx diry dirz wavelength polx poly polz weight"  
    elif qty == 'time_wavelength':
        qty = "time wavelength"  
    pass


    qtys = qty.split() if qty.find(' ')>-1 else [qty]
    nqty = len(qtys)

    if nqty == 1:    
        nr, nc = 1, 1
    elif nqty == 2:    
        nr, nc = 1, 2
    elif nqty == 9:    
        nr, nc = 3, 3
    elif nqty == 12:    
        nr, nc = 3, 4
    elif nqty == 4:    
        nr, nc = 2, 2
    else:    
        nr, nc = nqty//2, 2    # guessing
      

    legend = cfg.pop("legend")
    color = list(cfg.pop("color"))
    title = cfg.pop("title", None)
    path = cfg.pop("path", None)
    suptitle = cfg.pop("suptitle", None)

    fig = plt_figure(cfg)

    if not suptitle is None:
        fig.suptitle(suptitle, fontsize=14, fontweight='bold')

    for pl,qty in enumerate(qtys):
        plt.subplot(nr,nc,pl+1)
        plt.title(qty) 
        for i, ary in enumerate(arys):
            col = color[i] 
            cfg.update(label=getattr(ary,'label',col), color=col)
            val = getattr(ary, qty)
            plt.hist(val, **cfg)
        pass
        if type(legend) == list:
            assert len(legend) == len(qtys)
            ulegend = legend[pl]
        else:
            ulegend = legend
        if ulegend:
            plt.legend()
    pass

    plt_save(path)

    plt.show()






if __name__ == '__main__':
   g = geometry()
   print g

