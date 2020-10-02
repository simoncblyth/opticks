
Opticks In Production
======================

Embedded Opticks using G4OK package 
-------------------------------------

In production, Opticks is intended to be run in an embedded mode 
where, Geant4 and Opticks communicate via “gensteps” and “hits”.
This works via some Geant4 dependant glue code within each detectors simulation framework 
that does the below:

* inhibits CPU generation of optical photons from G4Scintillation and G4Cerenkov processes, 
  instead "gensteps" are collected

* invokes embedded Opticks (typically at the end of each event) 
  passing the collected "gensteps" across to Opticks which performs the 
  propagation 

* pulls back the PMT hits and populates standard Geant4 hit collections with these

Once the details of the above integration have been revisted for JUNO example 
integration code will be provided within the Opticks repository. 


