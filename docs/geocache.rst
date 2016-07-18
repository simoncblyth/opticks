Geometry Cache
===============

Parsing large XML geometry files can be timeconsuming.  In order to avoid this
repeatedly paying this expense all geometry data is serialized into NPY and txt files
within the geocache.  
These buffers can subsequently be loaded from file and directly copied to the GPU.

Setting IDPATH
-----------------

Each detector geometry or selection applied to a detector geometry has a 
separate geocache stored within the **IDPATH** directory.
Opticks launches end by outputting messages like the below::

    # geocache directory corresponding to OPTICKS_ARGS --dpib --tracer  
    export IDPATH=/usr/local/opticks/opticksdata/export/dpib/cfg4.d41d8cd98f00b204e9800998ecf8427e.dae  

    # geocache directory corresponding to OPTICKS_ARGS --jpmt --tracer  
    export IDPATH=/usr/local/opticks/opticksdata/export/juno/test3.fcc8b4dc9474af8826b29bf172452160.dae

    # geocache directory corresponding to OPTICKS_ARGS --dyb --tracer  
    export IDPATH=/usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae


The IDPATH identifies the geocache directory corresponding to the geometry selected by the 
arguments. Opticks python analysis scripts require the IDPATH to be set in order
to access geometry data.

Copy/paste the export lines into your .bash_profile prior to using python 
analysis scripts. 

:doc:`../ana/proplib` is a simple example of a python script accessing the geocache.

Structure of the geocache
---------------------------

All geometry data including material and surface properties 
is serialized in the geocache::

    simon:~ blyth$ cd /usr/local/opticks/opticksdata/export/juno/test3.fcc8b4dc9474af8826b29bf172452160.dae
    simon:test3.fcc8b4dc9474af8826b29bf172452160.dae blyth$ find . 
    ./GBndLib
    ./GBndLib/GBndLibIndex.npy
    ./GItemList
    ./GItemList/GMaterialLib.txt
    ./GItemList/GScintillatorLib.txt
    ./GItemList/GSourceLib.txt
    ./GItemList/GSurfaceLib.txt
    ./GMaterialLib
    ./GMaterialLib/GMaterialLib.npy
    ./GMergedMesh
    ./GMergedMesh/0
    ./GMergedMesh/0/aiidentity.npy
    ./GMergedMesh/0/bbox.npy
    ./GMergedMesh/0/boundaries.npy
    ./GMergedMesh/0/center_extent.npy
    ./GMergedMesh/0/colors.npy
    ./GMergedMesh/0/identity.npy
    ./GMergedMesh/0/iidentity.npy
    ./GMergedMesh/0/indices.npy
    ./GMergedMesh/0/itransforms.npy
    ./GMergedMesh/0/meshes.npy
    ./GMergedMesh/0/nodeinfo.npy
    ./GMergedMesh/0/nodes.npy
    ./GMergedMesh/0/normals.npy
    ./GMergedMesh/0/sensors.npy
    ./GMergedMesh/0/transforms.npy
    ./GMergedMesh/0/vertices.npy
    ... 
    ./GScintillatorLib
    ./GScintillatorLib/GScintillatorLib.npy
    ./GScintillatorLib/LS
    ./GScintillatorLib/LS/ABSLENGTH.npy
    ./GScintillatorLib/LS/AlphaFASTTIMECONSTANT.npy
    ./GScintillatorLib/LS/AlphaSLOWTIMECONSTANT.npy
    ./GScintillatorLib/LS/AlphaYIELDRATIO.npy
    ./GScintillatorLib/LS/FASTCOMPONENT.npy
    ./GScintillatorLib/LS/GammaFASTTIMECONSTANT.npy
    ./GScintillatorLib/LS/GammaSLOWTIMECONSTANT.npy
    ./GScintillatorLib/LS/GammaYIELDRATIO.npy
    ./GScintillatorLib/LS/NeutronFASTTIMECONSTANT.npy
    ./GScintillatorLib/LS/NeutronSLOWTIMECONSTANT.npy
    ./GScintillatorLib/LS/NeutronYIELDRATIO.npy
    ./GScintillatorLib/LS/RAYLEIGH.npy
    ./GScintillatorLib/LS/ReemissionFASTTIMECONSTANT.npy
    ./GScintillatorLib/LS/REEMISSIONPROB.npy
    ./GScintillatorLib/LS/ReemissionSLOWTIMECONSTANT.npy
    ./GScintillatorLib/LS/ReemissionYIELDRATIO.npy
    ./GScintillatorLib/LS/RESOLUTIONSCALE.npy
    ./GScintillatorLib/LS/RINDEX.npy
    ./GScintillatorLib/LS/SCINTILLATIONYIELD.npy
    ./GScintillatorLib/LS/SLOWCOMPONENT.npy
    ./GSourceLib
    ./GSourceLib/GSourceLib.npy
    ./GSurfaceLib
    ./GSurfaceLib/GSurfaceLib.npy
    ./GSurfaceLib/GSurfaceLibOptical.npy
    ./GTreePresent.txt
    ./MeshIndexLocal.json
    ./MeshIndexSource.json





