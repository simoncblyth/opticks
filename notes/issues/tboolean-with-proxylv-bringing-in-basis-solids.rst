tboolean-with-proxylv-bringing-in-basis-solids
=================================================

Context
----------

Following :doc:`tboolean-resurrection` added capability 
for python opticks.analytic.csg:CSG to use a *proxylv=lvIdx* argument 
causing the corresponding standard solid to be included from the 
basis GMeshLib (whica also houses the analytic NCSG).

From tboolean-box--::

 753 box = CSG("box3", param=[300,300,200,0], emit=0,  boundary="Vacuum///GlassSchottF2", proxylv=${PROXYLV:--1} )
 754 
 755 CSG.Serialize([container, box], args )


Doing this required adding GMesh+NCSG to GMeshLib persisting as described 
in :doc:`review-test-geometry` and handling the proxying in GGeoTest and GMaker.


Observations
---------------

* having to double run the compute and viz is a pain when proxying 


::


   PROXYLV=0 tboolean.sh proxy 
   PROXYLV=0 tbooleanviz.sh proxy 

      # large cylinder poking out the box,  
      # container auto-resizing not working ?


   PROXYLV=20 tboolean.sh
   PROXYLV=20 tbooleanviz.sh

      # 20-inch PMT shape
      # changing sheetmask from 0x1 to 0x2 to make +Z emissive rather that -Z not working 





container auto sizing not working with proxies
--------------------------------------------------


* done in NCSGList::load so not proxy aware




