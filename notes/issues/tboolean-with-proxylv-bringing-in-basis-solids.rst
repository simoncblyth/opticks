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

* having to double run the compute and viz is a pain when proxying : NOW FIXED
* the black time-zero moving into position glitch is distracting : MOVING START TIME TO ZERO IS AN OK WORKAROUND


PROXYLV=0 tboolean.sh 
   * large cylinder poking out the box,  
   * container auto-resizing not working ? NOW FIXED

PROXYLV=1 tboolean.sh 
   * cylinder with a hole
   * photons only in ring ? but black time zero viz glitch makes uncertain 
   * setting start time to zero rather than 0.2 avoids the glitch

PROXYLV=2 tboolean.sh 
   * cylinder with a hole
   * again photons in ring 

PROXYLV=3 tboolean.sh 
   * cylinder

PROXYLV=4 tboolean.sh 
PROXYLV=5 tboolean.sh 
   * thin beams, the black before time issue particularly clear 

PROXYLV=6 tboolean.sh 
   * thin plate 
PROXYLV=7 tboolean.sh 
   * beam shape 

PROXYLV=8 tboolean.sh
PROXYLV=9 tboolean.sh
   * large flat square plates

PROXYLV=10 tboolean.sh
   * thick plate with cyclindrical hole part way thru, CSG coincidence speckles apparent
 
PROXYLV=11 tboolean.sh
PROXYLV=12 tboolean.sh
   * squat box   

PROXYLV=13 tboolean.sh
   * sphere with small cylinder protrusion on top  

PROXYLV=14 tboolean.sh
   * sphere 

PROXYLV=15 tboolean.sh
   * vertical cylinder with hole
   * more normal photon behaviour with this smaller piece  

*PROXYLV=16 tboolean.sh*
   * *asserts with unexpected left transform*

PROXYLV=17 tboolean.sh
   * observatory dome, nice propagation

PROXYLV=18 tboolean.sh
   * cathode cap, nice propagation

PROXYLV=19 tboolean.sh
   * remainder with cap cut, nice propagation, but non-physical pre-emption evident
   *
   * REVISIT-AFTER-FIXES : non-physical pre-emption is gone  

PROXYLV=20 tboolean.sh
   * 20-inch PMT shape
   * changing sheetmask from 0x1 to 0x2 to make +Z emissive rather that -Z not working 
   *
   * similar pre-emption issue to 18 
   * REVISIT-AFTER-FIXES : non-physical pre-emption is gone  

PROXYLV=21 tboolean.sh
   * PMT shape

*PROXYLV=22 tboolean.sh* 
   * asserts, NOW FIXED : see below
   * cylinder

PROXYLV=23 tboolean.sh
   * 23   PMT_3inch_inner1_solid_ell_helper0x510ae30 ce0 0.0000,0.0000,14.5216,38.0000 ce1 0.0000,0.0000,0.0000,38.0000 23
   * looks like cathode cap, propagation looks ok
    
PROXYLV=24 tboolean.sh
   * 24   PMT_3inch_inner2_solid_ell_helper0x510af10 ce0 0.0000,0.0000,-4.4157,38.0000 ce1 0.0000,0.0000,0.0000,38.0000 24
   * fruit bowl, prop ok 

PROXYLV=25 tboolean.sh
   * 25 PMT_3inch_body_solid_ell_ell_helper0x510ada0 ce0 0.0000,0.0000,4.0627,40.0000 ce1 0.0000,0.0000,0.0000,40.0000 25
   * looks like ellipsoid with lower quarter chopped : prop ok, manages to make a star pattern "cusp" 
       
*PROXYLV=26 tboolean.sh*
*PROXYLV=27 tboolean.sh* 
   * 26                PMT_3inch_cntr_solid0x510afa0 ce0 0.0000,0.0000,-45.8740,29.9995 ce1 0.0000,0.0000,0.0000,29.9995 26
   * 27                 PMT_3inch_pmt_solid0x510aae0 ce0 0.0000,0.0000,-17.9373,57.9383 ce1 0.0000,0.0000,0.0000,57.9383 27
     
   * both assert, NOW FIXED : see below 
   * 26: cylinder 
   * 27: cylinder union with sphere : has some ox deviation in TO BT BR BR BT SA 

PROXYLV=28 tboolean.sh 
   * 28                     sChimneyAcrylic0x5b310c0 ce0 0.0000,0.0000,0.0000,520.0000 ce1 0.0000,0.0000,0.0000,520.0000 28 
   * thick vertical cylinder with large hole  
   * photons go thru the hole, so only scatters hit the thing : chi2 deviates, probably just low stats
   * TODO: arrange targetting to hit the thing 


PROXYLV=29 tboolean.sh 
   * 29                          sChimneyLS0x5b312e0 ce0 0.0000,0.0000,0.0000,1965.0000 ce1 0.0000,0.0000,0.0000,1965.0000 29
   * solid vertical cylinder
   * Curious the normal square of propagating photons has become a diffuse circular patch.
   * This presumably is a clue to the strange propagation visualizations seen with large solids compared to timemax.
     Maybe when photons fail to hit anything within the time domain they do not appear in the viz.
   
PROXYLV=30 tboolean.sh 
   * 30                       sChimneySteel0x5b314f0 ce0 0.0000,0.0000,0.0000,1665.0000 ce1 0.0000,0.0000,0.0000,1665.0000 30
   * vertical cylinder pipe
   
   * The normal square of propagating photons has become a diffuse ring.

PROXYLV=31 tboolean.sh 
   * 31                          sWaterTube0x5b30eb0 ce0 0.0000,0.0000,0.0000,1965.0000 ce1 0.0000,0.0000,0.0000,1965.0000 31
   * solid vertical cylinder
   
   * Again a diffuse circular patch.
    
   * Selecting the second most frequent history "TO SA" 0x8d 
     (ie photons that miss the solid and just sail to the container on other side) 
     and there is no visible propagation visualization.
    

PROXYLV=32 tboolean.sh 
PROXYLV=33 tboolean.sh 
   * 32                        svacSurftube0x5b3bf50 ce0 0.0000,0.0000,0.0000,4.0000 ce1 0.0000,0.0000,0.0000,4.0000 32
   * 33                           sSurftube0x5b3ab80 ce0 0.0000,0.0000,0.0000,5.0000 ce1 0.0000,0.0000,0.0000,5.0000 33
    
   * small boxes : these are my placeholders for the guidetube torii

PROXYLV=34 tboolean.sh 
   * 34                         sInnerWater0x4bd3660 ce0 0.0000,0.0000,850.0000,20900.0000 ce1 0.0000,0.0000,0.0000,20900.0000 34
   * very big sphere with protrusion at top 
   * Only "TO SC SA" 0x86d backscatters? have any visualization

PROXYLV=35 tboolean.sh 
   * 35                      sReflectorInCD0x4bd3040 ce0 0.0000,0.0000,849.0000,20901.0000 ce1 0.0000,0.0000,0.0000,20901.0000 35
   * very big sphere with protrusion at top 
   * ox : high fdisc ~0.20


PROXYLV=36 tboolean.sh 
   * 36                     sOuterWaterPool0x4bd2960 ce0 0.0000,0.0000,0.0000,21750.0000 ce1 0.0000,0.0000,0.0000,21750.0000 36
   * very big cylinder
   * Again only "TO SC SA" 0x86d have a visualization
     

*PROXYLV=37 tboolean.sh* 
*PROXYLV=38 tboolean.sh* 
    * 37                         sPoolLining0x4bd1eb0 ce0 0.0000,0.0000,-1.5000,21753.0000 ce1 0.0000,0.0000,0.0000,21753.0000 37
    * 38                         sBottomRock0x4bcd770 ce0 0.0000,0.0000,-1500.0000,24750.0000 ce1 0.0000,0.0000,0.0000,24750.0000 38
    
    * both assert, NOW FIXED : see below   
    * 37,38: very big cylinders


PROXYLV=39 tboolean.sh 
    * 39                              sWorld0x4bc2350 ce0 0.0000,0.0000,0.0000,60000.0000 ce1 0.0000,0.0000,0.0000,60000.0000 39
    * big box
    * Only "TO SC SA" 0x86d and "TO SC SC SA" 0x866d have a visualization 



CMaker::MakeSolid asserts for PROXYLV 16,22,26,27,37,38
---------------------------------------------------------------------------------------------

* :doc:`tboolean-proxylv-CMaker-MakeSolid-asserts`

* after implementing back translation from ncylinder to G4Polycone for non z-symmetric cylinders 
  the asserts of 22,26,27,37,38 are fixed leaving just the "temple" 16 (sFasteners)


try to viz and propagate together fails : the old linux interop chestnut OR not : its just hits buffer : FIXED with OEvent::downloadHitsInterop 
----------------------------------------------------------------------------------------------------------------------------------------------------

* :doc:`OEvent_downloadHits_fail_in_interop`



container auto sizing not working with proxies : FIXED by a refactor
-------------------------------------------------------------------------


* done in NCSGList::load so not proxy aware

* fixed by refactor of NCSGList GGeoTest 
  and additions to GMaker and GMeshMaker


event and animation timings need auto adjustment as change size of geometry
---------------------------------------------------------------------------------

* when *proxylv* pulls in a big piece of geometry the animation goes real slow 
  as the time ranges are setup for smaller geometry



making --interop trump --compute : FIXED by rejig of OpticksMode 
-----------------------------------------------------------------

After the fix the "--interop" will trump the "--compute" argument within tboolean.sh::

    PROXYLV=2 tboolean.sh proxy --cvd 1 --interop


Initial simple hasArg in Opticks::init correctly sets interop when have both "--interop" and "--compute" but then::

   2019-06-10 09:52:51.116 ERROR [404357] [OpticksViz::renderLoop@528] OpticksViz::renderLoop early exit due to InteractivityLevel 0


::

    087     m_interactivity(m_ok->getInteractivityLevel()),
    ...
    524 void OpticksViz::renderLoop()
    525 {
    526     if(m_interactivity == 0 )
    527     {
    528         LOG(LEVEL) << "early exit due to InteractivityLevel 0  " ;                       
    529         return ;
    530     }


Fix this by moving the mode decision into OpticksMode


analysis needs adjusting for proxy locations : FIXED 
----------------------------------------------------------

* :doc:`opticks-event-paths`


