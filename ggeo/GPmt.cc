#include <cstdio>
#include <cassert>
#include <cstring>
#include <climits>
#include <map>

// npy-
#include "NGLM.hpp"
#include "NPY.hpp"
#include "NSlice.hpp"

// okc-
#include "Opticks.hh"
#include "OpticksResource.hh"

// ggeo-
#include "GVector.hh"
#include "GItemList.hh"
#include "GBndLib.hh"
#include "GCSG.hh"
#include "GParts.hh"
#include "GPmt.hh"

#include "PLOG.hh"

const char* GPmt::FILENAME = "GPmt.npy" ;  
const char* GPmt::FILENAME_CSG = "GPmt_csg.npy" ;  


GPmt* GPmt::load(Opticks* ok, GBndLib* bndlib, unsigned int index, NSlice* slice)
{
    GPmt* pmt = NULL ; 
    OpticksResource* resource = ok->getResource();
    std::string path = resource->getPmtPath(index); 
    bool exists = OpticksResource::existsFile(path.c_str(), FILENAME) ;

    LOG(info) << "GPmt::load"
              << " index " << index 
              << " exists " << ( exists ? "Y" : "N" )
              << " path " << path 
              ;    

    if(exists)
    {
        pmt = new GPmt(ok, bndlib, index);
        pmt->loadFromCache(slice);
        pmt->setPath(path.c_str());
    }
    else
    {
        LOG(warning) << "GPmt::load resource does not exist " << path ;  
    }
    return pmt ; 
}

/**

simon:ggeo blyth$ l /usr/local/opticks/opticksdata/export/DayaBay/GPmt/0/
total 80
-rw-r--r--  1 blyth  staff   848 Jul  5  2016 GPmt.npy
-rw-r--r--  1 blyth  staff   289 Jul  5  2016 GPmt.txt
-rw-r--r--  1 blyth  staff   289 Jul  5  2016 GPmt_boundaries.txt
-rw-r--r--  1 blyth  staff   848 Jul  5  2016 GPmt_check.npy
-rw-r--r--  1 blyth  staff   289 Jul  5  2016 GPmt_check.txt
-rw-r--r--  1 blyth  staff  1168 Jul  5  2016 GPmt_csg.npy
-rw-r--r--  1 blyth  staff    47 Jul  5  2016 GPmt_csg.txt
-rw-r--r--  1 blyth  staff    74 Jul  5  2016 GPmt_lvnames.txt
-rw-r--r--  1 blyth  staff    47 Jul  5  2016 GPmt_materials.txt
-rw-r--r--  1 blyth  staff    74 Jul  5  2016 GPmt_pvnames.txt

**/


const unsigned GPmt::NTRAN = 3 ; 

void GPmt::loadFromCache(NSlice* slice)
{
    OpticksResource* resource = m_ok->getResource();

    //const char* base = resource->getIdPath();
    const char* base = resource->getDetectorBase();

    bool relative ;   
    std::string relpath = resource->getPmtPath(m_index, relative=true); 
    std::string path    = resource->getPmtPath(m_index, relative=false ); 

    LOG(verbose) << "GPmt::loadFromCache"
               << " base " << base
               << " relpath " << relpath 
               << " path " << path 
               ;


    GItemList*  bndSpec_orig = GItemList::load(base, "GPmt_boundaries", relpath.c_str() );
    GItemList*  materials = GItemList::load(base, "GPmt_materials", relpath.c_str() );
    GItemList*  lvnames = GItemList::load(base, "GPmt_lvnames", relpath.c_str() );
    GItemList*  pvnames = GItemList::load(base, "GPmt_pvnames", relpath.c_str() );

    NPY<float>* partBuf_orig = NPY<float>::load( path.c_str(), FILENAME );
    NPY<float>* csgBuf_orig = NPY<float>::load( path.c_str(), FILENAME_CSG );

    NPY<float>* csgBuf(NULL);
    NPY<float>* partBuf(NULL);
    GItemList* bndSpec(NULL);

    if(slice)
    {
        partBuf = partBuf_orig->make_slice(slice) ; 
        csgBuf = csgBuf_orig->make_slice(slice) ; 
        bndSpec = bndSpec_orig->make_slice(slice);
        LOG(info) << "GPmt::loadFromCache slicing partBuf " 
                  << " partBuf_orig " << partBuf_orig->getShapeString() 
                  << " partBuf " << partBuf->getShapeString()
                  ; 

    }
    else
    {
        partBuf = partBuf_orig ; 
        csgBuf = csgBuf_orig ; 
        bndSpec = bndSpec_orig ; 
    }


    LOG(info) << "GPmt::loadFromCache"
              << " bndSpec_orig " << bndSpec_orig
              << " bndSpec " << bndSpec
              << " materials " << materials
              << " lvnames " << lvnames
              << " pvnames " << pvnames
              << " slice " << slice 
              ;


    //if(bndSpec) bndSpec->dump("(GItemList)bndSpec->dump()"); 

    NPY<unsigned>* idxBuf = NPY<unsigned>::make(1,4);
    idxBuf->zero();

    NPY<float>* tranBuf = NPY<float>::make(0,NTRAN,4,4);
    tranBuf->zero();

    NPY<float>* planBuf = NPY<float>::make(0,4);
    planBuf->zero();

    GParts* parts = new GParts(idxBuf, partBuf, tranBuf, planBuf, bndSpec, m_bndlib);
    parts->setAnalyticVersion(getIndex());
    parts->setPrimFlag(CSG_FLAGPARTLIST);

    //parts->close() ; // apropos here ? NO : get OMAT-unset-error. Must setContainingMaterial before close ?

    const char* sensorSurface = m_ok->getSensorSurface();
    const char* medium = m_ok->getAnalyticPMTMedium() ; 

    parts->setContainingMaterial(medium);   
    parts->setSensorSurface(sensorSurface) ;
    parts->setPartList(); // setting primFlag to CSG_FLAGPARTLIST
    parts->close();      // registerBoundaries, makePrimBuffer


    setParts(parts);

    GCSG* csg = new GCSG(csgBuf, materials, lvnames, pvnames ) ;
    setCSG(csg);

    //GItemList* bndSpec2 = parts->getBndSpec();
    //if(bndSpec2) bndSpec2->dump("(GItemList)bndSpec2->dump()"); 
     
}


GPmt::GPmt(Opticks* ok, GBndLib* bndlib, unsigned int index) 
    :
    m_ok(ok),
    m_bndlib(bndlib),
    m_index(index),
    m_parts(NULL),
    m_csg(NULL),
    m_path(NULL)
{
}


void GPmt::dump(const char* msg)
{
    LOG(info) << msg 
              << " m_index " << m_index
              << " m_path " << ( m_path ? m_path : "-" ) 
              << " m_parts " << m_parts
              << " m_csg " << m_csg
              << " m_bndlib " << m_bndlib
              ;

    GParts* pts = getParts();
    LOG(fatal) << "GPmt.GParts::dump" ; 
    pts->dump("GParts::dump", 20 );

    /*
    GCSG* csg = getCSG();
    csg->dump("GCSG::dump");
    */ 

    LOG(fatal) << "GPmt.GParts::Summary" ; 
    pts->Summary(); 
}


void GPmt::setParts(GParts* parts)
{
    m_parts = parts ; 
}
GParts* GPmt::getParts()
{
    return m_parts ; 
}

void GPmt::setCSG(GCSG* csg)
{
    m_csg = csg ; 
}
GCSG* GPmt::getCSG()
{
    return m_csg ; 
}

void GPmt::setPath(const char* path)
{
    m_path = path ? strdup(path) : NULL  ; 
}
const char* GPmt::getPath()
{
    return m_path ; 
}
unsigned GPmt::getIndex()
{
    return m_index ; 
}



/*
hmm have to break connection to solids 
as uncoincident translation rejigs boundaries away 
from direct translation triangulated ones

former approach grouped parts into "solids"
based on nodeindex of the detdesc traverse
which are used in cu/hemi-pmt.cu as the OptiX primitives
each with its own BBox computed from the parts
when a BBox is entererd OptiX invokes the intersect 
which loops over the parts of that solid 

SolidBuffer provides part counts and offsets

    788 RT_PROGRAM void intersect(int primIdx)
    789 {
    790   const uint4& solid    = solidBuffer[primIdx];
    791   unsigned int numParts = solid.y ;
    792   const uint4& identity = identityBuffer[primIdx] ;
    ...
    796   for(unsigned int p=0 ; p < numParts ; p++)
    797   {
    798       unsigned int partIdx = solid.x + p ;
    799 
    800       quad q0, q1, q2, q3 ;
    801 
    802       q0.f = partBuffer[4*partIdx+0];


in new uncoincident approach have different boundaries 
for different parts of the same "solid"
so need to get boundary index from the partBuffer 
rather than the solid buffer

No matter which grouping is used identity and solid buffer need the same
shape and identity content needs to mimic the triangulated ones (other than mesh index).
Triangulated identity buffer hails from GVolume::getIdentity with guint4 per solid

* node index (useful)
* mesh index (not used yet? handy for mesh debug)
* boundary index (critical)
* sensor surface index (critical)

All these are setup in AssimpGGeo eg with GVolume::setSensor 
and GVolume::setBoundary

   897     NSensorList* sens = gg->getSensorList();
   898     NSensor* sensor = sens->findSensorForNode( nodeIndex );
   899     solid->setSensor( sensor );

Node index provides connection to sensor, so need to know
what the analytic cathode node index is.

So need to associate "analytic nodes" of the cathode 
with corresponding normal triangulated GNode 
in order to get the sensor index, the boundary indices 
can be ontained from GBndLib
by adding the bndspec.

Actually thinking ahead to instanced identity just need the sensor corresponding to each 
instance.
 

Which grouping to use ?
~~~~~~~~~~~~~~~~~~~~~~~~

Groupin by original solid is making less sense, 
maybe a 4 way primitive split makes more sense
arranged by Z position of the bboxen

    
tubs                      bottom                    face0                    face1
    
OM///Pyrex                OM///Pyrex               OM///Pyrex                OM///Pyrex
Pyrex///Vacuum            Pyrex///OpaqueVacuum     Pyrex/SENSOR//Bialkali    Pyrex/SENSOR//Bialkali 
Vacuum///OpaqueVacuum     OpaqueVacuum///Vacuum    Bialkali///Vacuum         Bialkali///Vacuum


               original 
               material
Part Sphere        Pyrex  pmt-hemi-face-glass_part_zright       [0, 0, 0] r: 131.0 sz:  0.0 BB      [ -84.54  -84.54  100.07]      [  84.54   84.54  131.  ] z 115.53 OUTERMATERIAL///Pyrex
Part Sphere        Pyrex  pmt-hemi-top-glass_part_zmiddle    [0, 0, 43.0] r: 102.0 sz:  0.0 BB      [-101.17 -101.17   56.  ]      [ 101.17  101.17  100.07] z  78.03 OUTERMATERIAL///Pyrex
Part Sphere        Pyrex    pmt-hemi-bot-glass_part_zleft    [0, 0, 69.0] r: 102.0 sz:  0.0 BB      [-101.17 -101.17  -23.84]      [ 101.17  101.17   56.  ] z  16.08 OUTERMATERIAL///Pyrex
Part   Tubs        Pyrex               pmt-hemi-base_part   [0, 0, -84.5] r: 42.25 sz:169.0 BB      [ -42.25  -42.25 -169.  ]         [ 42.25  42.25 -23.84] z -96.42 OUTERMATERIAL///Pyrex

Part Sphere       Vacuum    pmt-hemi-face-vac_part_zright       [0, 0, 0] r: 128.0 sz:  0.0 BB         [-82.29 -82.29  98.05]      [  82.29   82.29  128.  ] z 113.02 Pyrex/lvPmtHemiCathodeSensorSurface//Bialkali
Part Sphere       Vacuum    pmt-hemi-top-vac_part_zmiddle    [0, 0, 43.0] r:  99.0 sz:  0.0 BB         [-98.14 -98.14  56.  ]         [ 98.14  98.14  98.05] z  77.02 Pyrex/lvPmtHemiCathodeSensorSurface//Bialkali
Part Sphere       Vacuum      pmt-hemi-bot-vac_part_zleft    [0, 0, 69.0] r:  99.0 sz:  0.0 BB         [-98.14 -98.14 -21.89]         [ 98.14  98.14  56.  ] z  17.06 Pyrex///OpaqueVacuum
Part   Tubs       Vacuum           pmt-hemi-base-vac_part   [0, 0, -81.5] r: 39.25 sz:166.0 BB      [ -39.25  -39.25 -164.5 ]         [ 39.25  39.25 -21.89] z -93.19 Pyrex///Vacuum

Part Sphere     Bialkali       pmt-hemi-cathode-face_part       [0, 0, 0] r:127.95 sz:  0.0 BB         [-82.25 -82.25  98.01]      [  82.25   82.25  127.95] z 112.98 Bialkali///Vacuum
Part Sphere     Bialkali      pmt-hemi-cathode-belly_part    [0, 0, 43.0] r: 98.95 sz:  0.0 BB         [-98.09 -98.09  55.99]         [ 98.09  98.09  98.01] z  77.00 Bialkali///Vacuum
Part Sphere OpaqueVacuum                pmt-hemi-bot_part    [0, 0, 69.0] r:  98.0 sz:  0.0 BB         [-97.15 -97.15 -29.  ]         [ 97.15  97.15  56.13] z  13.57 OpaqueVacuum///Vacuum
Part   Tubs OpaqueVacuum             pmt-hemi-dynode_part   [0, 0, -81.5] r:  27.5 sz:166.0 BB         [ -27.5  -27.5 -164.5]            [ 27.5  27.5   1.5] z -81.50 Vacuum///OpaqueVacuum

    
*/


