tboolean-boxx-bizarre-propagation-when-torchconfig-source-outside-container
=============================================================================

Get bizarre propagations when torchconfig source position is outside 
the container.


::

    [blyth@localhost evtbase]$ tboolean-torchconfig | tr "_" "\n"
    type=disc
    photons=100000
    mode=fixpol
    polarization=1,1,0
    frame=-1
    transform=1.000,0.000,0.000,0.000,0.000,1.000,0.000,0.000,0.000,0.000,1.000,0.000,0.000,0.000,0.000,1.000
    source=0,0,599
    target=0,0,0
    time=0.0
    radius=300
    distance=200
    zenithazimuth=0,1,0,1
    material=Vacuum
    wavelength=500



Get a sane propagation only after adjusting the container to fit the source

* note different argument forms of box and box3
* have to use containerautosize=0 otherwise it gets shrunk 

  * added honouring of this in NCSGList 


::

    1192 tboolean-boxx-(){  $FUNCNAME- | python $* ; }
    1193 tboolean-boxx--(){ cat << EOP 
    1194 import logging
    1195 log = logging.getLogger(__name__)
    1196 from opticks.ana.main import opticks_main
    1197 from opticks.analytic.csg import CSG  
    1198 
    1199 args = opticks_main(csgname="${FUNCNAME/--}")
    1200 
    1201 CSG.kwa = dict(poly="IM",resolution=20, verbosity=0, ctrl=0, containerscale=3.0, containerautosize=0 )
    1202 
    1203 container = CSG("box", param=[0,0,0,600], boundary='Rock//perfectAbsorbSurface/Vacuum', container=1  ) 
    1204 
    1205 box = CSG("box3", param=[300,300,200,0],  boundary="Vacuum///GlassSchottF2"  )
    1206 
    1207 CSG.Serialize([container, box], args )
    1208 EOP
    1209 }
    1210 


