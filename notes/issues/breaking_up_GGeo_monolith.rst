
Breaking up GGeo monolith adiabatically
=========================================

* factor related features into a tightly focussed lib eg GPmtLib 
  with well defined dependency on other libs that is a resident of GGeo 

* get this operational, checking with unit and integration tests::

     opticks-t
     tests-t 

* because locality/lifecycle are almost the same, such changes
  are usually straightforward

* note that GScene and GGeo are both GGeoBase subclasses : some libs that 
  do not need ana variants are returned in common from GScene via GGeo calls 


* consider moving such shared libs that do not need tri/ana variants
  up to a higher level in OpticksHub 

* moving the lib to live elsewhere is much easier than 
  moving a bunch of methods, with less clear dependencies 

* first cluster functionality then migrate












