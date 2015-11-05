#include "GPmt.hh"
#include "GCache.hh"
#include "GVector.hh"

// npy-
#include "NPY.hpp"
#include "NLog.hpp"

#include <map>
#include <cstdio>
#include <cassert>
#include <climits>


const char* GPmt::FILENAME = "GPmt.npy" ;  
const char* GPmt::SPHERE_ = "Sphere" ;
const char* GPmt::TUBS_   = "Tubs" ;

const char* GPmt::TypeName(unsigned int typecode)
{
    switch(typecode)
    {
        case 1:return SPHERE_ ; break ;
        case 2:return TUBS_   ; break ;
    }
    return NULL ; 
}


GPmt* GPmt::load(GCache* cache, unsigned int index)
{
    GPmt* pmt = new GPmt(cache, index);
    pmt->loadFromCache();
    return pmt ; 
}

void GPmt::loadFromCache()
{
    std::string path = m_cache->getPmtPath(m_index); 
    NPY<float>* partBuf = NPY<float>::load( path.c_str(), FILENAME );
    setPartBuffer(partBuf);
    import();
}


unsigned int GPmt::getNumParts()
{
    return m_part_buffer ? m_part_buffer->getShape(0) : 0 ; 
}

unsigned int GPmt::getNumSolids()
{
    return m_solid_buffer ? m_solid_buffer->getShape(0) : 0 ; 
}


unsigned int GPmt::getUInt(unsigned int i, unsigned int j, unsigned int k)
{
    assert(i < getNumParts() );
    float* data = m_part_buffer->getValues();
    uif_t uif ; 
    uif.f = data[i*NJ*NK+j*NJ+k] ;
    return uif.u ; 
}

unsigned int GPmt::getNodeIndex(unsigned int part_index)
{
    return getUInt(part_index, NODEINDEX_J, NODEINDEX_K);
}
unsigned int GPmt::getTypeCode(unsigned int part_index)
{
    return getUInt(part_index, TYPECODE_J, TYPECODE_K);
}
unsigned int GPmt::getIndex(unsigned int part_index)
{
    return getUInt(part_index, INDEX_J, INDEX_K);
}
unsigned int GPmt::getParent(unsigned int part_index)
{
    return getUInt(part_index, PARENT_J, PARENT_K);
}
unsigned int GPmt::getFlags(unsigned int part_index)
{
    return getUInt(part_index, FLAGS_J, FLAGS_K);
}

const char* GPmt::getTypeName(unsigned int part_index)
{
    unsigned int code = getTypeCode(part_index);
    return GPmt::TypeName(code);
}

void GPmt::import()
{
    unsigned int numQuads = m_part_buffer->getNumItems(); // buffer reshaped (-1,4) in pmt-/tree.py  items are 

    unsigned int nmin(INT_MAX) ; 
    unsigned int nmax(0) ; 

    // count parts for each nodeindex
    for(unsigned int i=0; i < getNumParts() ; i++)
    {
        unsigned int nodeIndex = getNodeIndex(i);
        //printf("init %2u : %d \n", i, nodeIndex );
        m_parts_per_solid[nodeIndex] += 1 ; 

        if(nodeIndex < nmin) nmin = nodeIndex ; 
        if(nodeIndex > nmax) nmax = nodeIndex ; 
    }

    // with part slicing maybe relax contiguous ?
    assert(nmax - nmin == m_parts_per_solid.size() - 1); 

    unsigned int num_solids = m_parts_per_solid.size() ;
    guint4* solidinfo = new guint4[num_solids] ;

    typedef std::map<unsigned int, unsigned int> UU ; 

    unsigned int offset = 0 ; 
    unsigned int n = 0 ; 
    for(UU::const_iterator it=m_parts_per_solid.begin() ; it != m_parts_per_solid.end() ; it++)
    {
        unsigned int s = it->first ; 
        unsigned int snp = it->second ; 

        guint4& si = *(solidinfo+n) ;
        si.x = offset ; 
        si.y = snp ;
        si.z = s ; 
        si.w = 0 ; 

        //printf("si %2u %2u %2u %2u \n", si.x,si.y,si.z,si.w);
        offset += snp ; 
        n++ ; 
    }

    NPY<unsigned int>* buf = NPY<unsigned int>::make( num_solids, 4 );
    buf->setData((unsigned int*)solidinfo);

    setSolidBuffer(buf);
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
        printf(" part %2u : node %2u type %2u \n", i, getNodeIndex(i), getTypeCode(i) ); 
    }

}


void GPmt::dump(const char* msg)
{
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
       unsigned int pid = getParent(i);
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
              else if( j == PARENT_J && k == PARENT_K)
              {
                  assert( uif.u == pid );
                  printf(" %6u pid  ", uif.u );
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
          printf("\n");
       }   
       printf("\n");
    }   
}


/*
Former part slicing kludge from OGeo::makeAnalyticGeometry 
working around GBuffer shape inflexibility.
Due to this migrated from GBuffer to NPY.::

    GBuffer* partBuf_orig = mm->getAnalyticGeometryBuffer();  // getter causes loading on first call
    GBuffer* partBuf = partBuf_orig ; 
    NSlice* pslice = mm->getPartSlice();
    if(pslice)
    {
        LOG(info) << "OGeo::makeAnalyticGeometry part slicing the part buffer" ;
        unsigned int nelem = partBuf_orig->getNumElements();
        assert(nelem == 4 && "expecting quads");
        partBuf_orig->reshape(4*GPmt::QUADS_PER_ITEM); 
        partBuf = partBuf_orig->make_slice(pslice);   
        partBuf_orig->reshape(nelem);
        partBuf->reshape(nelem);
    }
*/
