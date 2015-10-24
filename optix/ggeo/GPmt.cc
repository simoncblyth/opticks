#include "GPmt.hh"
#include "GBuffer.hh"
#include "GVector.hh"


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



GPmt* GPmt::load(const char* path)
{
    GBuffer* buf = GBuffer::load<float>(path);
    return new GPmt(buf);    
}

unsigned int GPmt::getUInt(unsigned int i, unsigned int j, unsigned int k)
{
    assert(i < m_num_parts );
    float* data = (float*)m_part_buffer->getPointer();
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






void GPmt::init()
{
    unsigned int numQuads = m_part_buffer->getNumItems(); // buffer reshaped (-1,4) in pmt-/tree.py  items are 
    setNumParts( numQuads/QUADS_PER_ITEM ) ; 

    unsigned int nmin(INT_MAX) ; 
    unsigned int nmax(0) ; 

    // count parts for each nodeindex
    for(unsigned int i=0; i < m_num_parts; i++)
    {
        unsigned int nodeIndex = getNodeIndex(i);
        //printf("init %2u : %d \n", i, nodeIndex );
        m_parts_per_solid[nodeIndex] += 1 ; 

        if(nodeIndex < nmin) nmin = nodeIndex ; 
        if(nodeIndex > nmax) nmax = nodeIndex ; 
    }


    // with part slicing maybe relax contiguous ?
    assert(nmax - nmin == m_parts_per_solid.size() - 1); 

    setNumSolids(m_parts_per_solid.size()) ;

    guint4* solidinfo = new guint4[m_num_solids] ;


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

        printf("si %2u %2u %2u %2u \n", si.x,si.y,si.z,si.w);
        offset += snp ; 
        n++ ; 
    }

    

    unsigned int size = sizeof(guint4);
    assert(size == sizeof(unsigned int)*4 );
    GBuffer* buf = new GBuffer(size*m_num_solids, (void*)solidinfo, size, 4);
    setSolidBuffer(buf);
}

/*

In [87]: s = np.load("/tmp/hemi-pmt-solids.npy")

In [88]: s
Out[88]: 
array([[ 0,  4,  0,  0],
       [ 4,  4,  1,  0],
       [ 8,  2,  2,  0],
       [10,  1,  3,  0],
       [11,  1,  4,  0]], dtype=uint32)

*/



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
    GBuffer* buf = m_part_buffer ;  
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


