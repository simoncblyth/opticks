hit-checking-beneath-experiment-level
======================================


Need way to check hits and the G4Opticks API for getting them etc..  
without ascending to the level of the experiment.

Obvious place is g4ok/tests/G4OKTest.cc 

G4OKTest 
---------

* DONE: it now boots from cache

TODO:

* provide a way for G4Opticks to boot from cache 
* hmm running from cache means cannot provide origin sensor placements...


* can GDML aux info be used to plant default genstep target volume ? 

  * this will avoid the need to give command line arguments to pick the target volume







      
