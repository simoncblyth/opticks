#include "GPmt.hh"
#include "GBuffer.hh"

#include <map>
#include <cstdio>
#include <cassert>
#include <climits>

#include <boost/log/trivial.hpp>
#define LOG BOOST_LOG_TRIVIAL
// trace/debug/info/warning/error/fatal



typedef union 
{   
   unsigned int u ; 
   int i ; 
   float f ; 
}              uif_t ; 


GPmt* GPmt::load(const char* path)
{
    GBuffer* buf = GBuffer::load<float>(path);
    return new GPmt(buf);    
}

unsigned int GPmt::getUInt(unsigned int i, unsigned int j, unsigned int k)
{
    float* data = (float*)m_buffer->getPointer();
    uif_t uif ; 
    uif.f = data[i*NJ*NK+j*NJ+k] ;
    return uif.u ; 
}

unsigned int GPmt::getNodeIndex(unsigned int part_index)
{
    assert(part_index < m_num_parts );
    return getUInt(part_index, NODEINDEX_J, NODEINDEX_K);
}
unsigned int GPmt::getTypeCode(unsigned int part_index)
{
    assert(part_index < m_num_parts );
    return getUInt(part_index, TYPECODE_J, TYPECODE_K);
}


void GPmt::init()
{
    unsigned int numQuads = m_buffer->getNumItems(); // buffer reshaped (-1,4) in pmt-/tree.py  items are 
    setNumParts( numQuads/QUADS_PER_ITEM ) ; 

    unsigned int nmin(INT_MAX) ; 
    unsigned int nmax(0) ; 

    for(unsigned int i=0; i < m_num_parts; i++)
    {
        unsigned int nodeIndex = getNodeIndex(i);
        //printf("init %2u : %d \n", i, nodeIndex );
        m_parts_per_solid[nodeIndex] += 1 ; 

        if(nodeIndex < nmin) nmin = nodeIndex ; 
        if(nodeIndex > nmax) nmax = nodeIndex ; 
    }

    assert(nmin == 0 && nmax == m_parts_per_solid.size() - 1); // expecting contiguous node index starting at zero 
    setNumSolids(m_parts_per_solid.size()) ;
}


void GPmt::Summary(const char* msg)
{
    LOG(info) << msg 
              << " num_parts " << m_num_parts 
              << " num_solids " << m_num_solids
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

    for(unsigned int i=0 ; i < m_num_parts ; i++)
    {
        printf(" part %2u : node %2u type %2u \n", i, getNodeIndex(i), getTypeCode(i) ); 
    }

}


void GPmt::dump(const char* msg)
{
    GBuffer* buf = m_buffer ;  
    assert(buf);

    float* data = (float*)buf->getPointer();
    unsigned int numBytes    = buf->getNumBytes();
    unsigned int numQuads    = buf->getNumItems();
    unsigned int numElements = buf->getNumElements();
    unsigned int quadsPerItem = QUADS_PER_ITEM ; 
    unsigned int numItems    = numQuads/quadsPerItem ;  // buffer is reshaped for easy  GBuffer::load in pmt-/tree.py  
 
    LOG(info) << msg 
              << " numBytes " << numBytes 
              << " numQuads " << numQuads 
              << " quadsPerItem " << quadsPerItem 
              << " numItems " << numItems 
              << " numElements " << numElements
              << " numElements*numItems*sizeof(T) " << numElements*numItems*sizeof(float)  
              ;   

    assert(numElements < 17); // elements within an item, eg 3/4 for float3/float4  
    assert(numElements*numQuads*sizeof(float) == numBytes );  

    unsigned int ni = numItems ;
    assert( NJ*NK == quadsPerItem*numElements );  
    uif_t uif ; 

    for(unsigned int i=0; i < ni; i++)
    {   
       for(unsigned int j=0 ; j < NJ ; j++)
       {   
          for(unsigned int k=0 ; k < NK ; k++) 
          {   
              uif.f = data[i*NJ*NK+j*NJ+k] ;
              if( j == TYPECODE_J && k == TYPECODE_K )
                  printf(" %10d (typecode Sphere:1 Tubs:2) ", uif.i );
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


