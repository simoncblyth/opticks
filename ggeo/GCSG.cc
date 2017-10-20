#include "NGLM.hpp"
#include "NPY.hpp"

#include "GItemList.hh"
#include "GMaker.hh"
#include "GMergedMesh.hh"

#include "GCSG.hh"
#include "PLOG.hh"


// TODO: adopt OpticksCSG enum and names
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

GCSG::GCSG(NPY<float>* buffer, GItemList* materials, GItemList* lvnames, GItemList* pvnames) 
      :
      m_csg_buffer(buffer),
      m_materials(materials),
      m_lvnames(lvnames),
      m_pvnames(pvnames)
{
}
      
NPY<float>* GCSG::getCSGBuffer()
{
    return m_csg_buffer ; 
}

bool GCSG::isUnion(unsigned int i)
{
    return getTypeCode(i) == UNION ; 
}
bool GCSG::isIntersection(unsigned int i)
{
    return getTypeCode(i) == INTERSECTION ; 
}
bool GCSG::isSphere(unsigned int i)
{
    return getTypeCode(i) == SPHERE ; 
}
bool GCSG::isTubs(unsigned int i)
{
    return getTypeCode(i) == TUBS ; 
}



const char* GCSG::getTypeName(unsigned int i)
{
    unsigned int tc = getTypeCode(i);
    return TypeName(tc) ;
}


float GCSG::getX(unsigned int i){            return getFloat(i, 0, 0 ); }
float GCSG::getY(unsigned int i){            return getFloat(i, 0, 1 ); }
float GCSG::getZ(unsigned int i){            return getFloat(i, 0, 2 ); }
float GCSG::getOuterRadius(unsigned int i){  return getFloat(i, 0, 3 ); }

float GCSG::getStartTheta(unsigned int i) {  return getFloat(i, 1, 0 ); }
float GCSG::getDeltaTheta(unsigned int i) {  return getFloat(i, 1, 1 ); }
float GCSG::getSizeZ(unsigned int i) {       return getFloat(i, 1, 2 ); }
float GCSG::getInnerRadius(unsigned int i) { return getFloat(i, 1, 3 ); }

unsigned int GCSG::getTypeCode(unsigned int i){         return getUInt(i, 2, 0); }
unsigned int GCSG::getNodeIndex(unsigned int i) {       return getUInt(i, 2, 1); }
unsigned int GCSG::getParentIndex(unsigned int i) {     return getUInt(i, 2, 2); }
unsigned int GCSG::getSpare(unsigned int i) {           return getUInt(i, 2, 3); }

unsigned int GCSG::getIndex(unsigned int i) {           return getUInt(i, 3, 0); }
unsigned int GCSG::getNumChildren(unsigned int i) {     return getUInt(i, 3, 1); }
unsigned int GCSG::getFirstChildIndex(unsigned int i) { return getUInt(i, 3, 2); }
unsigned int GCSG::getLastChildIndex(unsigned int i) {  return getUInt(i, 3, 3); }



unsigned int GCSG::getUInt(unsigned int i, unsigned int j, unsigned int k)
{
    assert(i < getNumItems() );
    unsigned int l=0u ; 
    return m_csg_buffer->getUInt(i,j,k,l);
}

float GCSG::getFloat(unsigned int i, unsigned int j, unsigned int k)
{
    assert(i < getNumItems() );
    return m_csg_buffer->getValue(i,j,k);
}



unsigned int GCSG::getNumItems()
{
    if(!m_csg_buffer)
    {
        LOG(error) << "GCSG::getNumItems NULL buffer" ;
        return 0 ;  
    }
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




void GCSG::dump(const char* msg)
{
    LOG(info) << msg ; 

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




/*
GMergedMesh* GCSG::makeMergedMesh()
{
    GCSG* csg = this ; 

    GMergedMesh* mm = NULL ; 
    // follow general approach of CTestDetector::makePMT but using placeholder bbox 
    // to act as standin until have real CSG to triangles imp

    unsigned ni = csg->getNumItems(); 

    LOG(info) << "GCSG::makeMergedMesh"
              << " ni " << ni 
              ;

    for(unsigned i=0 ; i < ni ; i++)
    {
        unsigned int nix = csg->getNodeIndex(i); 
        if(nix == 0) continue ;
        // skip non-lv with nix:0, as those are constituents of the lv that get recursed over


        LOG(info) << "GCSG::makeMergedMesh"
                  << std::setw(4) << i 
                  << " nix " << std::setw(5) << nix 
                  ; 
    
    }
    return mm ; 
}
*/



