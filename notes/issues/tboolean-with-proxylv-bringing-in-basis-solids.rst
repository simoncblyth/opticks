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
* the black time-zero moving into position glitch is distracting 


::

   PROXYLV=0 tboolean.sh proxy --cvd 1 

      # large cylinder poking out the box,  
      # container auto-resizing not working ? NOW FIXED

   PROXYLV=1 tboolean.sh proxy
   PROXYLV=1 tbooleanviz.sh proxy

      # cylinder with a hole
      # photons only in ring ? but black time zero viz glitch makes uncertain 
      # setting start time to zero rather than 0.2 avoids the glitch

   PROXYLV=4 tboolean.sh proxy --cvd 1 
   PROXYLV=5 tboolean.sh proxy --cvd 1 
      # thin beams, the black before time issue particularly clear 

   PROXYLV=6 tboolean.sh proxy --cvd 1 
      # thin plate 

   PROXYLV=10 tboolean.sh proxy --cvd 1 
      # thick plate with cyclindrical hole part way thru, CSG coincidence speckles apparent
     
   PROXYLV=11 tboolean.sh proxy --cvd 1 
   PROXYLV=12 tboolean.sh proxy --cvd 1
      # squat box   

   PROXYLV=13 tboolean.sh proxy --cvd 1 
      # sphere with small cylinder protrusion on top  

   PROXYLV=14 tboolean.sh proxy --cvd 1 
      # sphere 

   PROXYLV=15 tboolean.sh proxy --cvd 1 
     # vertical cylinder with hole
     # more normal photon behaviour with this smaller piece  
     # perhaps issue is get absorption before long with large geometry ? but surely vacuum ?


   PROXYLV=16 tboolean.sh proxy --cvd 1 

    2019-06-09 23:16:02.989 FATAL [310360] [CTestDetector::makeChildVolume@135]  csg.spec Rock///Rock boundary 2 mother - lv UNIVERSE_LV pv UNIVERSE_PV mat Rock
    2019-06-09 23:16:02.989 INFO  [310360] [CTestDetector::makeDetector_NCSG@199]    0 spec Rock//perfectAbsorbSurface/Vacuum
    2019-06-09 23:16:02.989 FATAL [310360] [CTestDetector::makeChildVolume@135]  csg.spec Rock//perfectAbsorbSurface/Vacuum boundary 0 mother UNIVERSE_LV lv box_lv0_ pv box_pv0_ mat Vacuum
    2019-06-09 23:16:02.989 INFO  [310360] [CTestDetector::makeDetector_NCSG@199]    1 spec Vacuum///GlassSchottF2
    2019-06-09 23:16:02.990 FATAL [310360] [CMaker::MakeSolid_r@142]  unexpected non-identity left transform  depth 3 name un label un
    1.0000,0.0000,0.0000,0.0000 0.0000,1.0000,0.0000,0.0000 0.0000,0.0000,1.0000,0.0000 0.0000,125.0000,-70.0000,1.0000
    OKG4Test: /home/blyth/opticks/cfg4/CMaker.cc:150: static G4VSolid* CMaker::MakeSolid_r(const nnode*, unsigned int): Assertion `0' failed.


   PROXYLV=17 tboolean.sh proxy --cvd 1 
      # observatory dome, nice propagation

   PROXYLV=18 tboolean.sh proxy --cvd 1 
      # cathode cap, nice propagation

   PROXYLV=19 tboolean.sh proxy --cvd 1 
      # remainder with cap cut, nice propagation, but non-physical pre-emption evident


   PROXYLV=20 tboolean.sh proxy --cvd 1 

      # 20-inch PMT shape
      # changing sheetmask from 0x1 to 0x2 to make +Z emissive rather that -Z not working 
      #
      # similar pre-emption issue to 18 


    PROXYLV=22 tboolean.sh proxy --cvd 1 

    2019-06-09 23:29:06.619 INFO  [333016] [CTestDetector::makeDetector_NCSG@199]    1 spec Vacuum///GlassSchottF2
    OKG4Test: /home/blyth/opticks/cfg4/CMaker.cc:417: static G4VSolid* CMaker::ConvertPrimitive(const nnode*): Assertion `z2 > z1 && z2 == -z1' failed.
    /home/blyth/opticks/bin/o.sh: line 179: 333016 Aborted                 (core dumped) /home/blyth/local/opticks/lib/OKG4Test --okg4 --cvd 1 --envkey --rendermode +global,+axis --animtimemax 20 --timemax 20 --geocenter --stack 2180 --eye 1,0,0 --test --testconfig autoseqmap=TO:0,SR:1,SA:0_name=tboolean-proxy-22_outerfirst=1_analytic=1_csgpath=tboolean-proxy-22_mode=PyCsgInBox_autoobject=Vacuum/perfectSpecularSurface//GlassSchottF2_autoemitconfig=photons:600000,wavelength:380,time:0.2,posdelta:0.1,sheetmask:0x1,umin:0.45,umax:0.55,vmin:0.45,vmax:0.55,diffuse:1,ctmindiffuse:0.5,ctmaxdiffuse:1.0_autocontainer=Rock//perfectAbsorbSurface/Vacuum --torch --torchconfig type=disc_




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


analysis needs adjusting for proxy locations
------------------------------------------------

* :doc:`opticks-event-paths`


