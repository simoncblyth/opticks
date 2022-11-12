planned_code_reduction
=======================



The current active opticks packages are listed below with om-subs::

   epsilon:~ blyth$ om-subs
   okconf
   sysrap
   boostrap          ##
   npy               ##
   optickscore       ##
   ggeo              ##
   extg4             ##
   ana
   analytic
   bin
   CSG
   CSG_GGeo          ##
   GeoChain          ##
   qudarap
   gdxml
   u4
   CSGOptiX
   g4cx


The packages marked with "##" are planned to be removed.
They are currently only used during geometry translation, not 
during simulation.  


When I developed the CSG geometry model needed for OptiX 7 
I did so on top of the GGeo geometry model::

   .        extg4       CSG_GGeo
   Geant4  --->  GGeo   ---->   CSG

Which means that there is currently one too many full geometry models
in use with lots of code and tests.

So there is very stong code reduction potential.
As lots can go direct from Geant4 -> CSG.

Some intermediate geometry model is needed to do the geometry
factorization, I am developing that in a minimal way
in sysrap/stree.h u4/U4Tree.hh
Most of GGeo is not needed anymore.
The minimal approach is proving to make the translation
much faster.

So I can eliminate probably more that half of current Opticks code.
So the primary packages will become::

   sysrap   : base types, config
   qudarap  : QSim simulation implemented in CUDA (with no OptiX dependeny)
   u4       : Geant4 interface
   CSG      : geometry model
   CSGOptiX : accelerated CSG passing intersects to QSim
   g4cx     : top level interface





