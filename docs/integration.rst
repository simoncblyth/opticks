Opticks Integration with Geant4 Simulation
============================================

Geometry Overview
---------------------

Opticks is intended to be run from geometry  
exported from Geant4, so that you can have a 
single geometry source definition that is 
auto converted into an Opticks geometry.

For tests using simple geometries and photon sources 
it is also possible to define solids using a simple python CSG specification.


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


Simple Geometry Testing
~~~~~~~~~~~~~~~~~~~~~~~~~

For small scale Opticks debugging/testing it is convenient
to define geometry using a python CSG input representation.
On running the python scripts writes a serialization of the geometry 
to file which is read by Opticks.

Many examples can be seen in the `tboolean-` tests

* https://bitbucket.org/simoncblyth/opticks/src/tip/tests/tboolean.bash

You can also define simple optical photon sources via commandline **torch** config 
arguments. Following good experience with python defined test geometry it 
is planned to overhaul source configuation to use a similar approach.


Full Geometry Testing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Opticks has some functionality in the `Cfg4` package 
for comparing itself to a Geant4 simulation that it creates itself.  
This package however needs to revisited following Opticks recent 
move to fully analytic CSG geometries.





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
a geometry are configured by this script based on the command line arguments provided. 
This script together with `$OPTICKS_HOME/externals/opticksdata.bash` needs to be 
adapted for the new geometry.

Note it is possible via geometry query strings to define multiple virtual geometries 
that all use the same input geometry files selecting different volumes sets.


::

    simon:opticks blyth$ op --help
    ubin /usr/local/opticks/lib/OKTest cfm cmdline --help
    === op-export : OPTICKS_BINARY /usr/local/opticks/lib/OKTest

    op.sh : Opticks Launching Script
    ===================================

    The **op.sh** script launches different Opticks executables
    or scripts depending on the arguments provided. It also 
    sets environment variables picking a detector geometry
    and selecting volumes within the geometry.

    Most usage of Opticks should use this script.

    To see the options specific to particular scripts or
    executables use "-h" rather than the "--help" 
    that provides this text.


    Profile Setup 
    ---------------

    To save typing add the below bash function to your .bash_profile::

       op(){ op.sh $* ; }




    GEOMETRY SELECTION ARGUMENTS 

                    --dyb :                       DYB : DayaBay Near Site 
                   --dlin :                      DLIN : DayaBay LingAo Site 
                   --dfar :                      DFAR : DayaBay Far Site 
                   --dpib :                      DPIB : DayaBay PMT in Box of Mineral Oil Test Geometry 
                   --dsst :                      DSST : DYB debugging SST rib impingement 
                  --dsst2 :                     DSST2 : DYB debugging SST rib impingement 
                  --dlv65 :                     DLV65 : DYB cycybobo lvid 65  
                   --jpmt :                      JPMT : JUNO with PMTs 
                  --j1707 :                     J1707 : JUNO with ~36k 3inch PMTs, ~18k 20inch PMTs, torus guide tube  
                    --lxe :                       LXE : Geant4 LXe Liquid Xenon example 



    BINARY SELECTION ARGUMENTS 

                 --idpath :             OpticksIDPATH : Emit to stdout the path of the geocache directory for the geometry selected by arguments 
                   --keys :            InteractorKeys : List key controls available in GGeoViewTest  
                  --tcfg4 :                   CG4Test : Geant4 comparison simulation of simple test geometries  
                   --okg4 :                  OKG4Test : Integrated Geant4/Opticks runing allowing G4GUN steps to be directly Opticks GPU propagated.  
                 --tracer :               OTracerTest : Fast OpenGL viz and OptiX tracing, NO propagation. From ggeoview-/tests. Used for simple geometry/machinery checking 
              --gdml2gltf :              gdml2gltf.py : Once only geometry conversion of GDML input into GLTF file needed for analytic geocache creation 
                    --mat :          GMaterialLibTest : Dump properties of material identified by 0-based index , eg op --mat 0  
                   --cmat :          CMaterialLibTest :  
                   --surf :           GSurfaceLibTest : Dump properties of surface identified by 0-based index , eg op --surf 0  
                    --bnd :               GBndLibTest : Dump boundaries of a geometry, eg op --bnd --jpmt  
          --ctestdetector :         CTestDetectorTest : Test Geant4 simple detector construction using class cfg4-/CTestDetector  
          --cgdmldetector :         CGDMLDetectorTest : Test Geant4 GDML full detector construction using cfg4-/CGDMLDetector  
             --ngunconfig :            NGunConfigTest : Test Geant4 Gun configuration with npy-/NGunConfigTest  
    ...



Geocache
----------

Opticks makes extensive use of geocaching to avoid repetition of geometry 
processing. Expensive parsing of XML and multiple traversals of large geometry trees 
are done once only and resulting serialized NPY format buffers are written to
the geocache ready to be quickly loaded from file and then uploaded to the GPU.

The geocache allows Opticks initialization even with huge geometries to be kept 
to a few seconds only, facilitating fast iteration. 



