#include "GPmt.hh"
#include "GCache.hh"
#include "GVector.hh"
#include "GItemList.hh"
#include "GBndLib.hh"

// npy-
#include "NPY.hpp"
#include "NSlice.hpp"
#include "NLog.hpp"

#include <map>
#include <cstdio>
#include <cassert>
#include <climits>

const char* GPmt::CONTAINING_MATERIAL = "CONTAINING_MATERIAL" ;  
const char* GPmt::SENSOR_SURFACE = "SENSOR_SURFACE" ;  
const char* GPmt::FILENAME = "GPmt.npy" ;  
const char* GPmt::SPHERE_ = "Sphere" ;
const char* GPmt::TUBS_   = "Tubs" ;
const char* GPmt::BOX_    = "Box" ;

const char* GPmt::TypeName(unsigned int typecode)
{
    LOG(debug) << "GPmt::TypeName " << typecode ; 
    switch(typecode)
    {
        case SPHERE:return SPHERE_ ; break ;
        case   TUBS:return TUBS_   ; break ;
        case    BOX:return BOX_    ; break ;
        default:  assert(0) ; break ; 
    }
    return NULL ; 
}

GPmt* GPmt::load(GCache* cache, unsigned int index, NSlice* slice)
{
    GPmt* pmt = new GPmt(cache, index);
    pmt->loadFromCache(slice);
    return pmt ; 
}

void GPmt::loadFromCache(NSlice* slice)
{
    std::string relpath = m_cache->getPmtPath(m_index, true); 
    GItemList*  origSpec = GItemList::load(m_cache->getIdPath(), "GPmt", relpath.c_str() );

    std::string path = m_cache->getPmtPath(m_index); 
    NPY<float>* origBuf = NPY<float>::load( path.c_str(), FILENAME );


    NPY<float>* partBuf(NULL);
    GItemList* bndSpec(NULL);
    if(slice)
    {
        partBuf = origBuf->make_slice(slice) ; 
        bndSpec = origSpec->make_slice(slice);
        LOG(info) << "GPmt::loadFromCache slicing partBuf " 
                  << " origBuf " << origBuf->getShapeString() 
                  << " partBuf " << partBuf->getShapeString()
                  ; 

    }
    else
    {
        partBuf = origBuf ; 
        bndSpec = origSpec ; 
    }


    setBndSpec(bndSpec);
    setPartBuffer(partBuf);

    import();
}


void GPmt::setContainingMaterial(const char* material)
{
    m_bndspec->replaceField(0, GPmt::CONTAINING_MATERIAL, material );
}

void GPmt::setSensorSurface(const char* surface)
{
    m_bndspec->replaceField(1, GPmt::SENSOR_SURFACE, surface ) ; 
    m_bndspec->replaceField(2, GPmt::SENSOR_SURFACE, surface ) ; 
}



void GPmt::addContainer(gbbox& bb, const char* spec)
{
    unsigned int typecode = BOX ; 
    unsigned int nodeindex = getNumSolids() ; 
    unsigned int partindex = getNumParts() + 1  ; // 1-based  ?

    unsigned int boundary = m_bndlib->addBoundary(spec);
    const char* imat = m_bndlib->getInnerMaterialName(boundary);
    assert(strcmp(imat,"MineralOil")==0);

    LOG(info) << "GPmt::addContainer"
              << " nodeindex " << nodeindex 
              << " partindex " << partindex
              << " spec " << spec 
              << " boundary " << boundary
              << " inner material " << imat 
               ; 

    // fill in the blanks for CONTAINING_MATERIAL///Pyrex

    setContainingMaterial(imat); 


    NPY<float>* part = NPY<float>::make(1, NJ, NK );
    part->zero();

    assert(BBMIN_K == 0 );
    assert(BBMAX_K == 0 );
    unsigned int i = 0u ; 
    part->setQuad( i, BBMIN_J, bb.min.x, bb.min.y, bb.min.z , 0.f );
    part->setQuad( i, BBMAX_J, bb.max.x, bb.max.y, bb.max.z , 0.f );
    part->setUInt( i, NODEINDEX_J, NODEINDEX_K, nodeindex ); 
    part->setUInt( i, TYPECODE_J,  TYPECODE_K,  typecode ); 
    part->setUInt( i, INDEX_J,     INDEX_K,     partindex ); 
    part->setUInt( i, BOUNDARY_J,  BOUNDARY_K,  boundary ); 

    m_bndspec->add(spec);
    m_part_buffer->add(part);

    import(); // recreate the solid buffer
}



void GPmt::registerBoundaries()
{
   assert(m_bndlib); 
   unsigned int nbnd = m_bndspec->getNumKeys() ; 
   assert( getNumParts() == nbnd );
   for(unsigned int i=0 ; i < nbnd ; i++)
   {
       const char* spec = m_bndspec->getKey(i);
       unsigned int boundary = m_bndlib->addBoundary(spec);
       setBoundary(i, boundary);

      LOG(debug) << "GPmt::registerBoundaries " 
                << std::setw(3) << i
                << std::setw(30) << spec
                << " --> "
                << std::setw(4) << boundary 
                << std::setw(30) << m_bndlib->shortname(boundary)
                ;

   } 
}


unsigned int GPmt::getNumParts()
{
    return m_part_buffer ? m_part_buffer->getShape(0) : 0 ; 
}

unsigned int GPmt::getNumSolids()
{
    return m_solid_buffer ? m_solid_buffer->getShape(0) : 0 ; 
}

     
gfloat3 GPmt::getGfloat3(unsigned int i, unsigned int j, unsigned int k)
{
    float* data = m_part_buffer->getValues();
    float* ptr = data + i*NJ*NK+j*NJ+k ;
    return gfloat3( *ptr, *(ptr+1), *(ptr+2) ); 
}


guint4 GPmt::getSolidInfo(unsigned int isolid)
{
    unsigned int* data = m_solid_buffer->getValues();
    unsigned int* ptr = data + isolid*SK  ;
    return guint4( *ptr, *(ptr+1), *(ptr+2), *(ptr+3) );
}


gbbox GPmt::getBBox(unsigned int i)
{
   gfloat3 min = getGfloat3(i, BBMIN_J, BBMIN_K );  
   gfloat3 max = getGfloat3(i, BBMAX_J, BBMAX_K );  
   gbbox bb(min, max) ; 
   return bb ; 
}


unsigned int GPmt::getUInt(unsigned int i, unsigned int j, unsigned int k)
{
    assert(i < getNumParts() );
    return m_part_buffer->getUInt(i,j,k);

}

void GPmt::setUInt(unsigned int i, unsigned int j, unsigned int k, unsigned int value)
{
    assert(i < getNumParts() );
    m_part_buffer->setUInt(i,j,k, value);
}


unsigned int GPmt::getNodeIndex(unsigned int part)
{
    return getUInt(part, NODEINDEX_J, NODEINDEX_K);
}
unsigned int GPmt::getTypeCode(unsigned int part)
{
    return getUInt(part, TYPECODE_J, TYPECODE_K);
}
unsigned int GPmt::getIndex(unsigned int part)
{
    return getUInt(part, INDEX_J, INDEX_K);
}
unsigned int GPmt::getBoundary(unsigned int part)
{
    return getUInt(part, BOUNDARY_J, BOUNDARY_K);
}
void GPmt::setBoundary(unsigned int part, unsigned int boundary)
{
    setUInt(part, BOUNDARY_J, BOUNDARY_K, boundary);
}

std::string GPmt::getBoundaryName(unsigned int part)
{
    unsigned int boundary = getBoundary(part);
    std::string name = m_bndlib ? m_bndlib->shortname(boundary) : "" ;
    return name ;
}

unsigned int GPmt::getFlags(unsigned int part)
{
    return getUInt(part, FLAGS_J, FLAGS_K);
}


const char* GPmt::getTypeName(unsigned int part_index)
{
    unsigned int code = getTypeCode(part_index);
    return GPmt::TypeName(code);
}

void GPmt::import()
{
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
Triangulated identity buffer hails from GSolid::getIdentity with guint4 per solid

* node index (useful)
* mesh index (not used yet? handy for mesh debug)
* boundary index (critical)
* sensor surface index (critical)

All these are setup in AssimpGGeo eg with GSolid::setSensor 
and GSolid::setBoundary

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

    m_parts_per_solid.clear();
    unsigned int nmin(INT_MAX) ; 
    unsigned int nmax(0) ; 

    // count parts for each nodeindex
    for(unsigned int i=0; i < getNumParts() ; i++)
    {
        unsigned int nodeIndex = getNodeIndex(i);

        LOG(debug) << "GPmt::import"
                   << " i " << std::setw(3) << i  
                   << " nodeIndex " << std::setw(3) << nodeIndex
                   ;  
                     
        m_parts_per_solid[nodeIndex] += 1 ; 

        if(nodeIndex < nmin) nmin = nodeIndex ; 
        if(nodeIndex > nmax) nmax = nodeIndex ; 
    }

    unsigned int num_solids = m_parts_per_solid.size() ;
    //assert(nmax - nmin == num_solids - 1);  // expect contiguous node indices
    if(nmax - nmin != num_solids - 1)
    {
        LOG(warning) << "GPmt::import non-contiguous node indices"
                     << " nmin " << nmin 
                     << " nmax " << nmax
                     << " num_solids " << num_solids 
                     ; 
    }


    guint4* solidinfo = new guint4[num_solids] ;

    typedef std::map<unsigned int, unsigned int> UU ; 
    unsigned int part_offset = 0 ; 
    unsigned int n = 0 ; 
    for(UU::const_iterator it=m_parts_per_solid.begin() ; it != m_parts_per_solid.end() ; it++)
    {
        unsigned int node_index = it->first ; 
        unsigned int parts_for_solid = it->second ; 

        guint4& si = *(solidinfo+n) ;

        si.x = part_offset ; 
        si.y = parts_for_solid ;
        si.z = node_index ; 
        si.w = 0 ;              

        LOG(debug) << "GPmt::import solidinfo " << si.description() ;       

        part_offset += parts_for_solid ; 
        n++ ; 
    }

    NPY<unsigned int>* buf = NPY<unsigned int>::make( num_solids, 4 );
    buf->setData((unsigned int*)solidinfo);
    delete [] solidinfo ; 

    setSolidBuffer(buf);
}


void GPmt::dumpSolidInfo(const char* msg)
{
    LOG(info) << msg << " (part_offset, parts_for_solid, solid_index, 0) " ;
    for(unsigned int i=0 ; i < getNumSolids(); i++)
    {
        guint4 si = getSolidInfo(i);
        LOG(info) << si.description() ;
    }
}


void GPmt::Summary(const char* msg)
{
    LOG(info) << msg 
              << " num_parts " << getNumParts() 
              << " num_solids " << getNumSolids()
              ;
 
    typedef std::map<unsigned int, unsigned int> UU ; 
    for(UU::const_iterator it=m_parts_per_solid.begin() ; it!=m_parts_per_solid.end() ; it++)
    {
        unsigned int solid_index = it->first ; 
        unsigned int nparts = it->second ; 
        unsigned int nparts2 = getSolidNumParts(solid_index) ; 
        printf("%2u : %2u \n", solid_index, nparts );
        assert( nparts == nparts2 );
    }

    for(unsigned int i=0 ; i < getNumParts() ; i++)
    {
        std::string bn = getBoundaryName(i);
        printf(" part %2u : node %2u type %2u boundary [%3u] %s  \n", i, getNodeIndex(i), getTypeCode(i), getBoundary(i), bn.c_str() ); 
    }
}



void GPmt::dump(const char* msg)
{
    dumpSolidInfo(msg);

    NPY<float>* buf = m_part_buffer ; 
    assert(buf);
    assert(buf->getDimensions() == 3);

    unsigned int ni = buf->getShape(0) ;
    unsigned int nj = buf->getShape(1) ;
    unsigned int nk = buf->getShape(2) ;

    assert( nj == NJ );
    assert( nk == NK );

    float* data = buf->getValues();

    uif_t uif ; 

    for(unsigned int i=0; i < ni; i++)
    {   
       unsigned int tc = getTypeCode(i);
       unsigned int id = getIndex(i);
       unsigned int bnd = getBoundary(i);
       std::string  bn = getBoundaryName(i);
       unsigned int flg = getFlags(i);
       const char*  tn = getTypeName(i);

       for(unsigned int j=0 ; j < NJ ; j++)
       {   
          for(unsigned int k=0 ; k < NK ; k++) 
          {   
              uif.f = data[i*NJ*NK+j*NJ+k] ;
              if( j == TYPECODE_J && k == TYPECODE_K )
              {
                  assert( uif.u == tc );
                  printf(" %10u (%s) ", uif.u, tn );
              } 
              else if( j == INDEX_J && k == INDEX_K)
              {
                  assert( uif.u == id );
                  printf(" %6u id   " , uif.u );
              }
              else if( j == BOUNDARY_J && k == BOUNDARY_K)
              {
                  assert( uif.u == bnd );
                  printf(" %6u bnd  ", uif.u );
              }
              else if( j == FLAGS_J && k == FLAGS_K)
              {
                  assert( uif.u == flg );
                  printf(" %6u flg  ", uif.u );
              }
              else if( j == NODEINDEX_J && k == NODEINDEX_K)
                  printf(" %10d (nodeIndex) ", uif.i );
              else
                  printf(" %10.4f ", uif.f );
          }   

          if( j == BOUNDARY_J ) printf(" bn %s ", bn.c_str() );
          printf("\n");
       }   
       printf("\n");
    }   

}


