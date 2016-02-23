#include "GCSG.hh"
#include "GItemList.hh"
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

const char* GCSG::getMaterialName(unsigned int nodeindex)
{
    assert(m_materials);
    return m_materials->getKey(nodeindex) ;
}

const char* GCSG::getLVName(unsigned int nodeindex)
{
    assert(m_lvnames);
    return m_lvnames->getKey(nodeindex) ;
}

const char* GCSG::getPVName(unsigned int nodeindex)
{
    assert(m_pvnames);
    return m_pvnames->getKey(nodeindex) ;
}





float GCSG::getX(unsigned int i)
{
    return m_csg_buffer->getValue(i, 0, 0 );
}
float GCSG::getY(unsigned int i)
{
    return m_csg_buffer->getValue(i, 0, 1 );
}
float GCSG::getZ(unsigned int i)
{
    return m_csg_buffer->getValue(i, 0, 2 );
}
float GCSG::getOuterRadius(unsigned int i)
{
    return m_csg_buffer->getValue(i, 0, 3 );
}
float GCSG::getSizeZ(unsigned int i)
{
    return m_csg_buffer->getValue(i, 0, 3 );
}


float GCSG::getInnerRadius(unsigned int i)
{
    return m_csg_buffer->getValue(i, 1, 3 );
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
       unsigned int ix = getNodeIndex(i);
       unsigned int pr = getParentIndex(i);

       const char* mat = ix > 0 ? getMaterialName(ix - 1) : "" ; 
       const char* lvn = ix > 0 ? getLVName(ix - 1) : "" ; 
       const char* pvn = ix > 0 ? getPVName(ix - 1) : "" ; 

       printf(" ix %3d id %3d pr %3d nc %3d fc %3d lc %3d tc %3d tn %s mat %s lvn %s pvn %s  \n", ix, id, pr, nc, fc, lc, tc, tn, mat, lvn, pvn );  

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




