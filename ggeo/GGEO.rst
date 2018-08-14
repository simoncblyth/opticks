GGEO
======

GGeo
    top level holder of geometry libs
GScene
    used for analytic geometry : phasing out, now analytic is integrated within GGeo 

GVolume
    analogue of Geant4 PV, used to represent the volume tree 
GNode
    base class of GVolume, has a GMesh constituent
GNodeLib
    collects PV LV names

GGeoGLTF
    GGeo helper that can write a geometry to glTF 2.0 files
GSolidRec
    struct used by GGeoGLTF to record instances related to single solids 

GInstancer
    invoked by GGeo::prepareMeshes : finds instanced geometry and creates GMergedMesh for 
    each instance assembly and for the global non-instanced geometry.
GTree
    static helper methods used by GInstancer and GMergedMesh for serializing instance transforms and identity 
GTreePresent  
    creates text dumps of volume trees, made more readable by eliding large numbers of siblings 

GGeoBase
    protocol base class, subclasses include : GGeo, GScene, GGeoTest 
GBndLib
GMaterialLib
GScintillatorLib
GSurfaceLib
GSourceLib
    property libs

GPropertyLib
    base class for GBndLib, GMaterialLib, GScintillatorLib, GSurfaceLib, GSourceLib

GMeshLib
    holder of GMesh for each solid 
GMesh
    vertices and triangles of a solid, obtained from Geant4 polgonization
GBuffer
    used by GMesh to hold vertices, triangles etc.. : aiming to replace with NPY 

GGeoLib
    holder of GMergedMesh, one for global geometry and one for each instance assembly
GMergedMesh
    combination of GMesh  

GProperty
    domain and value arrays GAry holding a property as a function of wavelength 
GPropertyMap
    collection of GProperty : base for GMaterial, GSkinSurface, GBorderSurface, GSource
GMaterial
GSkinSurface
GSource
GBorderSurface
    property maps 
GOpticalSurface
    constituent of GPropertyMap used for GSkinSurface and GBorderSurface
GBnd
    collection of four indices of materials and surfaces representing a boundary 
GAry
    used for domain and value of GProperty
GDomain
    represents wavelength raster

GParts
    holder of analytic geometry, created from NCSG shapes, 
    provides combination and serialization 


Others
--------

::

    GArray
    GBBox
    GBBoxMesh
    GCIE
    GCSG
    GColorMap
    GColorizer
    GConstant
    GDrawable
    GEnums
    GGeoCfg
    GGeoSensor
    GGeoTest
    GIds
    GItemIndex
    GItemList
    GMaker
    GMatrix
    GMeshFixer
    GMeshMaker
    GPmt
    GPmtLib
    GSurfaceIndex
    GTransforms
    GTraverse
    GVector
    GVolumeList


