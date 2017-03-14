#include <map>
#include <iomanip>
#include <cstdio>
#include <cassert>
#include <climits>


#include "OpticksCSG.h"

// npy-
#include "NGLM.hpp"
#include "NPY.hpp"
#include "NSlice.hpp"
#include "NPart.hpp"
#include "NQuad.hpp"
#include "GLMFormat.hpp"

#include "GVector.hh"
#include "GItemList.hh"
#include "GBndLib.hh"
#include "GParts.hh"

#include "PLOG.hh"


const char* GParts::CONTAINING_MATERIAL = "CONTAINING_MATERIAL" ;  
const char* GParts::SENSOR_SURFACE = "SENSOR_SURFACE" ;  


GParts* GParts::combine(std::vector<GParts*> subs)
{
    // Concatenate vector of GParts instances into a single GParts instance

    GParts* parts = new GParts(); 
    GBndLib* bndlib = NULL ; 
    for(unsigned int i=0 ; i < subs.size() ; i++)
    {
        GParts* sp = subs[i];
        parts->add(sp);
        if(!bndlib) bndlib = sp->getBndLib(); 
    } 
    if(bndlib) parts->setBndLib(bndlib);
    return parts ; 
}


GParts* GParts::make(const npart& pt, const char* spec)
{
    // Serialize the npart shape (BOX, SPHERE or PRISM) into a (1,4,4) parts buffer. 
    // Then instanciate a GParts instance to hold the parts buffer 
    // together with the boundary spec.

    NPY<float>* part = NPY<float>::make(1, NJ, NK );
    part->zero();

    part->setQuad( pt.q0.f, 0u, 0u );
    part->setQuad( pt.q1.f, 0u, 1u );
    part->setQuad( pt.q2.f, 0u, 2u );
    part->setQuad( pt.q3.f, 0u, 3u );

    GParts* gpt = new GParts(part, spec) ;

    unsigned typecode = gpt->getTypeCode(0u);
    assert(typecode == CSG_BOX || typecode == CSG_SPHERE || typecode == CSG_PRISM);

    return gpt ; 
}

GParts* GParts::make(OpticksCSG_t csgflag, glm::vec4& param, const char* spec)
{
    float size = param.w ;  
    gbbox bb(gfloat3(-size), gfloat3(size));  

    if(csgflag == CSG_ZSPHERE)
    {
        bb.min.z = param.x*param.w ; 
        bb.max.z = param.y*param.w ; 
    } 

    NPY<float>* part = NPY<float>::make(1, NJ, NK );
    part->zero();

    assert(BBMIN_K == 0 );
    assert(BBMAX_K == 0 );

    unsigned int i = 0u ; 
    part->setQuad( i, PARAM_J, param.x, param.y, param.z, param.w );
    part->setQuad( i, BBMIN_J, bb.min.x, bb.min.y, bb.min.z , 0.f );
    part->setQuad( i, BBMAX_J, bb.max.x, bb.max.y, bb.max.z , 0.f );

    GParts* pt = new GParts(part, spec) ;

    pt->setTypeCode(0u, csgflag);
    //pt->setFlags(0u, csgflag);

    return pt ; 
} 



GParts::GParts(GBndLib* bndlib) 
      :
      m_part_buffer(NULL),
      m_bndspec(NULL),
      m_bndlib(bndlib),
      m_name(NULL),
      m_prim_buffer(NULL),
      m_closed(false),
      m_verbose(false)
{
      init() ; 
}

GParts::GParts(NPY<float>* buffer, const char* spec, GBndLib* bndlib) 
      :
      m_part_buffer(buffer),
      m_bndspec(NULL),
      m_bndlib(bndlib),
      m_prim_buffer(NULL),
      m_closed(false)
{
      init(spec) ; 
}
      
GParts::GParts(NPY<float>* buffer, GItemList* spec, GBndLib* bndlib) 
      :
      m_part_buffer(buffer),
      m_bndspec(spec),
      m_bndlib(bndlib),
      m_prim_buffer(NULL),
      m_closed(false)
{
      init() ; 
}


void GParts::save(const char* dir)
{
    if(!dir) return ; 

    const char* name = getName();    
    if(!name)
    {
        LOG(warning) << "GParts::save SET NAME BEFORE save" ;
        return ;  
    }

    LOG(info) << "GParts::save " << name << " to " << dir ; 

    if(m_part_buffer)
    {
        m_part_buffer->save(dir, name, "partBuffer.npy");    
    }
    if(m_prim_buffer)
    {
        m_prim_buffer->save(dir, name, "primBuffer.npy");    
    }
}


void GParts::setName(const char* name)
{
    m_name = name ? strdup(name) : NULL  ; 
}
const char* GParts::getName()
{
    return m_name ; 
}




void GParts::setVerbose(bool verbose)
{
    m_verbose = verbose ; 
}

bool GParts::isClosed()
{
    return m_closed ; 
}



unsigned int GParts::getPrimNumParts(unsigned int prim_index)
{
    return m_parts_per_prim.count(prim_index)==1 ? m_parts_per_prim[prim_index] : 0 ; 
}


void GParts::setBndSpec(GItemList* bndspec)
{
    m_bndspec = bndspec ;
}
GItemList* GParts::getBndSpec()
{
    return m_bndspec ; 
}

void GParts::setBndLib(GBndLib* bndlib)
{
    m_bndlib = bndlib ; 
}
GBndLib* GParts::getBndLib()
{
    return m_bndlib ; 
}


void GParts::setPrimBuffer(NPY<unsigned int>* prim_buffer)
{
    m_prim_buffer = prim_buffer ; 
}
NPY<unsigned int>* GParts::getPrimBuffer()
{
    return m_prim_buffer ; 
}

void GParts::setPartBuffer(NPY<float>* part_buffer)
{
    m_part_buffer = part_buffer ; 
}
NPY<float>* GParts::getPartBuffer()
{
    return m_part_buffer ; 
}




void GParts::init(const char* spec)
{
    m_bndspec = new GItemList("GParts");
    m_bndspec->add(spec);
    init();
}
void GParts::init()
{
    if(m_part_buffer == NULL && m_bndspec == NULL)
    {
        LOG(trace) << "GParts::init creating empty part_buffer and bndspec " ; 

        NPY<float>* empty = NPY<float>::make(0, NJ, NK );
        empty->zero();

        m_part_buffer = empty ; 
        m_bndspec = new GItemList("GParts");
    } 

    LOG(debug) << "GParts::init" 
              << " part_buffer items " << m_part_buffer->getNumItems() 
              << " bndspec items " <<  m_bndspec->getNumItems() 
              ;

    assert(m_part_buffer->getNumItems() == m_bndspec->getNumItems() );
}

unsigned int GParts::getNumParts()
{
    if(!m_part_buffer)
    {
        LOG(error) << "GParts::getNumParts NULL part_buffer" ; 
        return 0 ; 
    }

    assert(m_part_buffer->getNumItems() == m_bndspec->getNumItems() );
    return m_part_buffer->getNumItems() ;
}

void GParts::add(GParts* other)
{
    unsigned int n0 = getNumParts();
    //unsigned int nextPartIndex = n0 > 0 ? getIndex(n0 - 1) : 0 ;
    //assert(lastPartIndex == n0 - 1 ); 

    m_bndspec->add(other->getBndSpec());
    m_part_buffer->add(other->getPartBuffer());

    unsigned int n1 = getNumParts();
    for(unsigned int p=n0 ; p < n1 ; p++)
    {
        setIndex(p, p);
    }

    LOG(debug) << "GParts::add"
              << " n0 " << n0  
              << " n1 " << n1
              ;  
}



void GParts::setContainingMaterial(const char* material)
{
    // for flexibility persisted GParts should leave the outer containing material
    // set to a default marker name, to allow the GParts to be placed within other geometry

    m_bndspec->replaceField(0, GParts::CONTAINING_MATERIAL, material );
}

void GParts::setSensorSurface(const char* surface)
{
    m_bndspec->replaceField(1, GParts::SENSOR_SURFACE, surface ) ; 
    m_bndspec->replaceField(2, GParts::SENSOR_SURFACE, surface ) ; 
}



void GParts::close()
{
    registerBoundaries();
    makePrimBuffer(); 
    dumpPrimBuffer(); 
}

void GParts::registerBoundaries()
{
   assert(m_bndlib); 
   unsigned int nbnd = m_bndspec->getNumKeys() ; 
   assert( getNumParts() == nbnd );
   for(unsigned int i=0 ; i < nbnd ; i++)
   {
       const char* spec = m_bndspec->getKey(i);
       unsigned int boundary = m_bndlib->addBoundary(spec);
       setBoundary(i, boundary);

       if(m_verbose)
       LOG(info) << "GParts::registerBoundaries " 
                << std::setw(3) << i 
                << " " << std::setw(30) << spec
                << " --> "
                << std::setw(4) << boundary 
                << " " << std::setw(30) << m_bndlib->shortname(boundary)
                ;

   } 
}


void GParts::makePrimBuffer()
{
    // Derives prim buffer from the parts buffer
    //
    // * flag from the first part of each nodeIndex is promoted into primitive buffer  
    //
    // "prim" here was previously named "solid"
    // but thats confusing due to other solids so renamed
    // to "prim" as this corresponds to OptiX primitives GPU side, 
    // see oxrap/cu/hemi-pmt.cu::intersect
    //
    // "parts" are associated to their containing "prim" 
    // via the NodeIndex property  
    // so to prep a boolean composite need to:
    //
    // * arrange for constituent parts to share the same NodeIndex 
    // * set intersect/union/difference flag in parts buffer for first part
    //
    //   ^^^^^^^ TODO: generalize for CSG tree 
    //

    m_parts_per_prim.clear();
    m_flag_prim.clear();

    unsigned int nmin(INT_MAX) ; 
    unsigned int nmax(0) ; 

    unsigned numParts = getNumParts() ; 

    LOG(info) << "GParts::makePrimBuffer"
              << " numParts " << numParts 
              ;
 

    // count parts for each nodeindex
    for(unsigned int i=0; i < numParts ; i++)
    {
        unsigned int nodeIndex = getNodeIndex(i);
        //unsigned int flg = getFlags(i);
        //std::string msk = ShapeMask(flg);
        unsigned typ = getTypeCode(i);
        std::string  typName = CSGName((OpticksCSG_t)typ);
 
        LOG(info) << "GParts::makePrimBuffer"
                   << " i " << std::setw(3) << i  
                   << " nodeIndex " << std::setw(3) << nodeIndex
                   << " typName " << typName 
                   ;  
                     
        m_parts_per_prim[nodeIndex] += 1 ; 

        // flag from the first part of each nodeIndex is promoted into primitive buffer 
        if(m_flag_prim.count(nodeIndex) == 0) m_flag_prim[nodeIndex] = typ ; 

        if(nodeIndex < nmin) nmin = nodeIndex ; 
        if(nodeIndex > nmax) nmax = nodeIndex ; 
    }

    unsigned int num_prim = m_parts_per_prim.size() ;
    //assert(nmax - nmin == num_solids - 1);  // expect contiguous node indices
    if(nmax - nmin != num_prim - 1)
    {
        LOG(warning) << "GParts::makePrimBuffer non-contiguous node indices"
                     << " nmin " << nmin 
                     << " nmax " << nmax
                     << " num_prim " << num_prim
                     ; 
    }


    guint4* priminfo = new guint4[num_prim] ;

    typedef std::map<unsigned int, unsigned int> UU ; 
    unsigned int part_offset = 0 ; 
    unsigned int n = 0 ; 
    for(UU::const_iterator it=m_parts_per_prim.begin() ; it != m_parts_per_prim.end() ; it++)
    {
        unsigned int node_index = it->first ; 
        unsigned int parts_for_prim = it->second ; 
        unsigned int flg_for_prim = m_flag_prim[node_index] ; 

        guint4& pri = *(priminfo+n) ;

        pri.x = part_offset ; 
        pri.y = parts_for_prim ;
        pri.z = node_index ; 
        pri.w = flg_for_prim ;            // <--- prim/boolean-opcode ?  

        LOG(info) << "GParts::makePrimBuffer priminfo " << pri.description() ;       

        part_offset += parts_for_prim ; 
        n++ ; 
    }

    NPY<unsigned int>* buf = NPY<unsigned int>::make( num_prim, 4 );
    buf->setData((unsigned int*)priminfo);
    delete [] priminfo ; 

    setPrimBuffer(buf);
}



void GParts::dumpPrim(unsigned primIdx)
{
    // following access pattern of oxrap/cu/hemi-pmt.cu::intersect

    NPY<unsigned int>* primBuffer = getPrimBuffer();
    NPY<float>*        partBuffer = getPartBuffer();

    if(!primBuffer) return ; 
    if(!partBuffer) return ; 

    glm::uvec4 prim = primBuffer->getQuadU(primIdx) ;

    unsigned partOffset = prim.x ; 
    unsigned numParts   = prim.y ; 
    unsigned primFlags  = prim.w ; 

    LOG(info) << " primIdx "    << std::setw(3) << primIdx 
              << " partOffset " << std::setw(3) << partOffset 
              << " numParts "   << std::setw(3) << numParts
              << " primFlags "  << std::setw(5) << primFlags 
              << " CSGName "  << CSGName((OpticksCSG_t)primFlags) 
              << " prim "       << gformat(prim)
              ;

    for(unsigned int p=0 ; p < numParts ; p++)
    {
        unsigned int partIdx = partOffset + p ;

        nquad q0, q1, q2, q3 ;

        q0.f = partBuffer->getVQuad(partIdx,0);  
        q1.f = partBuffer->getVQuad(partIdx,1);  
        q2.f = partBuffer->getVQuad(partIdx,2);  
        q3.f = partBuffer->getVQuad(partIdx,3);  

        NPart_t partType = (NPart_t)q2.i.w ;

        LOG(info) << " p " << std::setw(3) << p 
                  << " partIdx " << std::setw(3) << partIdx
                  << " partType " << partType
                  << " partName " << PartName(partType)
                  ;

    }


}


void GParts::dumpPrimBuffer(const char* msg)
{
    NPY<unsigned int>* primBuffer = getPrimBuffer();
    NPY<float>*        partBuffer = getPartBuffer();
    LOG(info) << msg ; 
    if(!primBuffer) return ; 
    if(!partBuffer) return ; 

    LOG(info) 
        << " primBuffer " << primBuffer->getShapeString() 
        << " partBuffer " << partBuffer->getShapeString() 
        ; 

    { 
        unsigned ni = primBuffer->getShape(0) ; 
        unsigned nj = primBuffer->getShape(1) ; 
        unsigned nk = primBuffer->getShape(2) ; 
        assert( ni > 0 && nj == 4 && nk == 0 );
    }

    { 
        unsigned ni = partBuffer->getShape(0) ; 
        unsigned nj = partBuffer->getShape(1) ; 
        unsigned nk = partBuffer->getShape(2) ; 
        assert( ni > 0 && nj == 4 && nk == 4 );
    }


    for(unsigned primIdx=0 ; primIdx < primBuffer->getShape(0) ; primIdx++) dumpPrim(primIdx);
}


void GParts::dumpPrimInfo(const char* msg)
{
    unsigned int numPrim = getNumPrim() ;
    LOG(info) << msg << " (part_offset, parts_for_prim, prim_index, prim_flags) numPrim:" << numPrim  ;

    for(unsigned int i=0 ; i < numPrim ; i++)
    {
        guint4 pri = getPrimInfo(i);
        LOG(info) << pri.description() ;
    }
}


void GParts::Summary(const char* msg)
{
    LOG(info) << msg 
              << " num_parts " << getNumParts() 
              << " num_prim " << getNumPrim()
              ;
 
    typedef std::map<unsigned int, unsigned int> UU ; 
    for(UU::const_iterator it=m_parts_per_prim.begin() ; it!=m_parts_per_prim.end() ; it++)
    {
        unsigned int prim_index = it->first ; 
        unsigned int nparts = it->second ; 
        unsigned int nparts2 = getPrimNumParts(prim_index) ; 
        printf("%2u : %2u \n", prim_index, nparts );
        assert( nparts == nparts2 );
    }

    for(unsigned int i=0 ; i < getNumParts() ; i++)
    {
        std::string bn = getBoundaryName(i);
        printf(" part %2u : node %2u type %2u boundary [%3u] %s  \n", i, getNodeIndex(i), getTypeCode(i), getBoundary(i), bn.c_str() ); 
    }
}




unsigned int GParts::getNumPrim()
{
    return m_prim_buffer ? m_prim_buffer->getShape(0) : 0 ; 
}
const char* GParts::getTypeName(unsigned int part_index)
{
    unsigned int code = getTypeCode(part_index);
    //return GParts::TypeName(code);
    return PartName((NPart_t)code);


}
     
float* GParts::getValues(unsigned int i, unsigned int j, unsigned int k)
{
    float* data = m_part_buffer->getValues();
    float* ptr = data + i*NJ*NK+j*NJ+k ;
    return ptr ; 
}
     
gfloat3 GParts::getGfloat3(unsigned int i, unsigned int j, unsigned int k)
{
    float* ptr = getValues(i,j,k);
    return gfloat3( *ptr, *(ptr+1), *(ptr+2) ); 
}
guint4 GParts::getPrimInfo(unsigned int iprim)
{
    unsigned int* data = m_prim_buffer->getValues();
    unsigned int* ptr = data + iprim*SK  ;
    return guint4( *ptr, *(ptr+1), *(ptr+2), *(ptr+3) );
}
gbbox GParts::getBBox(unsigned int i)
{
   gfloat3 min = getGfloat3(i, BBMIN_J, BBMIN_K );  
   gfloat3 max = getGfloat3(i, BBMAX_J, BBMAX_K );  
   gbbox bb(min, max) ; 
   return bb ; 
}

void GParts::enlargeBBoxAll(float epsilon)
{
   for(unsigned int part=0 ; part < getNumParts() ; part++) enlargeBBox(part, epsilon);
}

void GParts::enlargeBBox(unsigned int part, float epsilon)
{
    float* pmin = getValues(part,BBMIN_J,BBMIN_K);
    float* pmax = getValues(part,BBMAX_J,BBMAX_K);

    glm::vec3 min = glm::make_vec3(pmin) - glm::vec3(epsilon);
    glm::vec3 max = glm::make_vec3(pmax) + glm::vec3(epsilon);
 
    *(pmin+0 ) = min.x ; 
    *(pmin+1 ) = min.y ; 
    *(pmin+2 ) = min.z ; 

    *(pmax+0 ) = max.x ; 
    *(pmax+1 ) = max.y ; 
    *(pmax+2 ) = max.z ; 

    LOG(debug) << "GParts::enlargeBBox"
              << " part " << part 
              << " epsilon " << epsilon
              << " min " << gformat(min) 
              << " max " << gformat(max)
              ; 

}


unsigned int GParts::getUInt(unsigned int i, unsigned int j, unsigned int k)
{
    assert(i < getNumParts() );
    unsigned int l=0u ; 
    return m_part_buffer->getUInt(i,j,k,l);
}
void GParts::setUInt(unsigned int i, unsigned int j, unsigned int k, unsigned int value)
{
    assert(i < getNumParts() );
    unsigned int l=0u ; 
    m_part_buffer->setUInt(i,j,k,l, value);
}



unsigned int GParts::getNodeIndex(unsigned int part)
{
    return getUInt(part, NODEINDEX_J, NODEINDEX_K);
}

unsigned int GParts::getTypeCode(unsigned int part)
{
    return getUInt(part, TYPECODE_J, TYPECODE_K);
}



unsigned int GParts::getIndex(unsigned int part)
{
    return getUInt(part, INDEX_J, INDEX_K);
}
unsigned int GParts::getBoundary(unsigned int part)
{
    return getUInt(part, BOUNDARY_J, BOUNDARY_K);
}


void GParts::setNodeIndex(unsigned int part, unsigned int nodeindex)
{
    setUInt(part, NODEINDEX_J, NODEINDEX_K, nodeindex);
}

void GParts::setTypeCode(unsigned int part, unsigned int typecode)
{
    setUInt(part, TYPECODE_J, TYPECODE_K, typecode);
}


void GParts::setIndex(unsigned int part, unsigned int index)
{
    setUInt(part, INDEX_J, INDEX_K, index);
}
void GParts::setBoundary(unsigned int part, unsigned int boundary)
{
    setUInt(part, BOUNDARY_J, BOUNDARY_K, boundary);
}


/*
unsigned int GParts::getFlags(unsigned int part)
{
    return getUInt(part, FLAGS_J, FLAGS_K);
}
void GParts::setFlags(unsigned int part, unsigned int flags)
{
    setUInt(part, FLAGS_J, FLAGS_K, flags);
}
void GParts::setFlagsAll(unsigned int flags)
{
    for(unsigned int i=0 ; i < getNumParts() ; i++) setFlags(i, flags);
}
*/


void GParts::setBoundaryAll(unsigned int boundary)
{
    for(unsigned int i=0 ; i < getNumParts() ; i++) setBoundary(i, boundary);
}





std::string GParts::getBoundaryName(unsigned int part)
{
    unsigned int boundary = getBoundary(part);
    std::string name = m_bndlib ? m_bndlib->shortname(boundary) : "" ;
    return name ;
}




void GParts::dump(const char* msg)
{
    LOG(info) << "GParts::dump " << msg ; 

    dumpPrimInfo(msg);

    NPY<float>* buf = m_part_buffer ; 
    assert(buf);
    assert(buf->getDimensions() == 3);

    unsigned int ni = buf->getShape(0) ;
    unsigned int nj = buf->getShape(1) ;
    unsigned int nk = buf->getShape(2) ;

    LOG(info) << "GParts::dump ni " << ni ; 

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

       //unsigned int flg = getFlags(i);
       //std::string msk = ShapeMask(flg);

       std::string csg = CSGName((OpticksCSG_t)tc);

       const char*  tn = getTypeName(i);

       for(unsigned int j=0 ; j < NJ ; j++)
       {   
          for(unsigned int k=0 ; k < NK ; k++) 
          {   
              uif.f = data[i*NJ*NK+j*NJ+k] ;
              if( j == TYPECODE_J && k == TYPECODE_K )
              {
                  assert( uif.u == tc );
                  printf(" %10u (%s) TYPECODE ", uif.u, tn );
              } 
              else if( j == INDEX_J && k == INDEX_K)
              {
                  assert( uif.u == id );
                  printf(" %6u <-id   " , uif.u );
              }
              else if( j == BOUNDARY_J && k == BOUNDARY_K)
              {
                  assert( uif.u == bnd );
                  printf(" %6u <-bnd  ", uif.u );
              }
              else if( j == FLAGS_J && k == FLAGS_K)
              {
                  assert( uif.u == flg );
                  printf(" %6u <-flg CSG %s ", uif.u, csg.c_str() );
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


