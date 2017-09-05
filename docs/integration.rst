Opticks Integration with Geant4 Simulation
============================================

Geometry Integration : Exporting Geant4 geometry into G4DAE and GDML files
------------------------------------------------------------------------------

Opticks typically operates from geometries exported from 
Geant4 using the standard GDML format exporter as well 
as the G4DAE exporter which exports COLLADA format .dae files
(an XML based industry standard 3D file format with triangulated mesh geometry)
with extra information including all optical material and surface properties.  

* https://bitbucket.org/simoncblyth/g4dae

The code required to export G4DAE and GDML is very similar using 
the G4DAEParser from *g4dae* and the Geant4 standard G4GDMLParser.
These exports should be done simultaneously (from the same process) 
so that memory address names match.

::

    G4VPhysicalVolume* world_pv = m_detector->Construct();
    G4DAEParser* g4dae = new G4DAEParser ;
    G4bool refs = true ;
    G4bool recreatePoly = false ; 
    G4int nodeIndex = -1 ;   // so World is volume 0 
    g4dae->Write(path, world_pv, refs, recreatePoly, nodeIndex );


Future of Opticks Geometry 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The recent(mid 2017) addition of GPU CSG intersection capabilities to Opticks
gives the possibility to operate from standard Geant4 GDML geometry files alone
in future. However GDML files do not currently contain all the 
wavelength dependant material and surface optical properties, that 
are included within G4DAE files.  
The Geant4 polygonization included within G4DAE files is 
useful for OpenGL visualizations within Opticks, although  
Opticks has some nascent capabilities at generating polygonizations 
these are not yet reliable.


Event Data Integration : Getting Photons into Opticks  
----------------------------------------------------------

The Geant4 processes `G4Cerenkov` and `G4Scintillation` generate large numbers 
of optical photons on the CPU. Opticks moves this optical photon generation 
to the GPU via modifications of the code for these processes. 

The photon generation loop of these processes is replaced by 
collection of so called **genstep** parameters of the generation. 
These gensteps are serialized into a buffer and copied to the GPU where 
ported versions of the inner photon generation loops are run

* https://bitbucket.org/simoncblyth/opticks/src/tip/optixrap/cu/cerenkovstep.h
* https://bitbucket.org/simoncblyth/opticks/src/tip/optixrap/cu/scintillationstep.h

In effect the inner loop of the photon generation is moved to the GPU.  
Although this generation is not time consuming, the copying of large numbers of photons 
from the CPU to the GPU and duplication of the photon memory allocation on 
CPU and GPU can be avoided by generating the photons on the GPU.


Outdated Chroma Live Integration Example
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* https://bitbucket.org/simoncblyth/env/src/tip/chroma/G4DAEChroma/src/

This stage was previously implemented in the context of 
the Chroma package.  As Chroma was implemented in python a 
ZeroMQ networking approach was used to effect the integration to Geant4, 
sending photons or **gensteps** over the network to be GPU propagated.

Planned Opticks Integration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

As Opticks is implemented in C++ a simple single process 
approach to integration is planned to be implemented.  
Prior to live integration Opticks testing currently uses **gensteps** 
persisted to file from Geant4. 

Detailed instructions appropriate to 
current Opticks await these stages from being revisited. 


Returning Photon Detector Hits to Geant4 Hit Collections
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* https://bitbucket.org/simoncblyth/env/src/tip/chroma/G4DAEChroma/src/

Again this stage was previously implemented in the context of Chroma 
in a somewhat (Daya Bay) experiment specific manner.


Integration Collaboration
-----------------------------

Example Daya Bay and JUNO Experiment geometry and **genstep** files 
are included in the OpticksData repository.

* https://bitbucket.org/simoncblyth/opticksdata/commits/all
* https://bitbucket.org/simoncblyth/opticksdata/src

A clone of this repository is done as part of the Opticks installation, eg 

::

    simon:~ blyth$ l /usr/local/opticks/opticksdata/
    total 8
    drwxr-xr-x  12 blyth  staff   408 Jul 22 10:07 export
    drwxr-xr-x   5 blyth  staff   170 Jul 22 10:07 gensteps
    drwxr-xr-x   3 blyth  staff   102 Jun 14 13:13 config
    -rw-r--r--   1 blyth  staff  1150 Jun 14 13:13 opticksdata.bash
    drwxr-xr-x   3 blyth  staff   102 Jun 14 13:13 refractiveindex
    drwxr-xr-x   4 blyth  staff   136 Jun 14 13:13 resource


The easiest way to collaborate on getting new geometries operational 
within Opticks is to follow the above export instructions to produce  
the below two files and commit them into the **opticksdata** repository.
If you do not yet have commit access a pull request can be made from 
a bitbucket fork of **opticksdata**.

::

    simon:~ blyth$ du -h /usr/local/opticks/opticksdata/export/juno1707/g4_00.{dae,gdml}
     24M    /usr/local/opticks/opticksdata/export/juno1707/g4_00.dae
     20M    /usr/local/opticks/opticksdata/export/juno1707/g4_00.gdml


Note that if your experiment has confidentiality concerns regarding 
sharing of an actual geometry you can collaborate/learn with an 
older/simplified demo version of your geometry.


Configuring Opticks to use the Geometry
--------------------------------------------

The **op.sh** script is the primary entry point to running Opticks executables, 
the specific executable to run and environment variables to setup specifying 
a geometry are configured via command line arguments to this script. This script 
together with `$OPTICKS_HOME/externals/opticksdata.bash` needs to be 
adapted for the new geometry.

Note it is possible via geometry query strings to define multiple virtual geometries 
that all use the same input geometry files selecting different volumes sets.


Geocache
----------

Opticks makes extensive use of geocaching to avoid repetition of geometry 
processing. Expensive parsing of XML and multiple traversals of large geometry trees 
are done once only and resulting serialized NPY format buffers are written to
the geocache ready to be quickly loaded from file and then uploaded to the GPU.

The geocache allows Opticks initialization even with huge geometries to be kept 
to a few seconds only, facilitating fast iteration. 



