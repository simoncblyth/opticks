Getting Started with Opticks
==============================

Exporting Geometry 
-------------------

If your geometry is modelled in Geant4 then it is straightforward to export it 
using my geometry exporter https://bitbucket.org/simoncblyth/g4dae
into the DAE file format used by Opticks. 
This works analogously to the GDML exporter.

::

     G4VPhysicalVolume* world_pv = m_detector->Construct();
     G4DAEParser* g4dae = new G4DAEParser ;
     G4bool refs = true ;
     G4bool recreatePoly = false ; 
     G4int nodeIndex = -1 ;   // so World is volume 0 
     g4dae->Write(path, world_pv, refs, recreatePoly, nodeIndex );


Exporting Event Data
-----------------------

Getting the event data is currently a little more involved, requiring the G4Cerenkov and G4Scintillation
processes to be modified. I intend to streamline this in future, to reduce the amount of code that
needs to be duplicated by each user. Examples for Daya Bay:

* http://dayabay.ihep.ac.cn/tracs/dybsvn/browser/dybgaudi/trunk/Simulation/DetSimChroma/src/DsChromaG4Scintillation.cc#L579
* http://dayabay.ihep.ac.cn/tracs/dybsvn/browser/dybgaudi/trunk/Simulation/DetSimChroma/src/DsChromaG4Cerenkov.cc#L308  

To use the resulting geometry and event files and do GPU simulations and visualizations
you need to install Opticks and dependencies : which require an NVIDIA GPU with
at least CUDA compute capability 3.0 (any NVIDIA GPU less than ~3 years old should be OK).

If this setup would take a long time, you could also share the geometry and event files with 
me to make visualisations.






