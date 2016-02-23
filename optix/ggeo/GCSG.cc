#include "GCSG.hh"
#include "NPY.hpp"
#include "NLog.hpp"

const char* GCSG::SPHERE_ = "Sphere" ;
const char* GCSG::TUBS_   = "Tubs" ;
const char* GCSG::UNION_    = "Union" ;
const char* GCSG::INTERSECTION_  = "Intersection" ;

const char* GCSG::TypeName(unsigned int typecode)
{
    LOG(debug) << "GCSG::TypeName " << typecode ; 
    switch(typecode)
    {
        case SPHERE:return SPHERE_ ; break ;
        case   TUBS:return TUBS_   ; break ;
        case  UNION:return UNION_    ; break ;
        case  INTERSECTION:return INTERSECTION_  ; break ;
        default:  assert(0) ; break ; 
    }
    return NULL ; 
}

unsigned int GCSG::getUInt(unsigned int i, unsigned int j, unsigned int k)
{
    assert(i < getNumItems() );
    unsigned int l=0u ; 
    return m_csg_buffer->getUInt(i,j,k,l);
}

unsigned int GCSG::getNumItems()
{
    return m_csg_buffer->getNumItems() ;
}

void GCSG::dump(const char* msg)
{
    NPY<float>* buf = m_csg_buffer ; 
    assert(buf);
    assert(buf->getDimensions() == 3);

    unsigned int ni = buf->getShape(0) ;
    unsigned int nj = buf->getShape(1) ;
    unsigned int nk = buf->getShape(2) ;

    assert( ni == getNumItems() );
    assert( nj == NJ );
    assert( nk == NK );

    float* data = buf->getValues();

    uif_t uif ; 

    for(unsigned int i=0; i < ni; i++)
    {   
       const char*  tn = getTypeName(i);
       unsigned int tc = getTypeCode(i);
       unsigned int id = getIndex(i);
       unsigned int nc = getNumChildren(i);
       unsigned int fc = getFirstChildIndex(i);
       unsigned int lc = getLastChildIndex(i);

       printf(" id %3d nc %3d fc %3d lc %3d tc %3d : %s \n", id, nc, fc, lc, tc, tn );  

       for(unsigned int j=0 ; j < NJ ; j++)
       {   
          for(unsigned int k=0 ; k < NK ; k++) 
          {   
              uif.f = data[i*NJ*NK+j*NJ+k] ;
              printf(" %10.4f ", uif.f );
          }   
          printf("\n");
       }   
       printf("\n");
    }   

}


