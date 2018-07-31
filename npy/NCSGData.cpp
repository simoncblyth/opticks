#include "SSys.hh"
#include "PLOG.hh"

#include "BFile.hh"
#include "BStr.hh"

#include "NPart.h"

#include "NPYBase.hpp"
#include "NPYList.hpp"
#include "NPYMeta.hpp"

#include "NPYSpec.hpp"
#include "NPYSpecList.hpp"
#include "NPY.hpp"
#include "NGLMExt.hpp"
#include "NCSGData.hpp"
#include "GLMFormat.hpp"

#define TREE_NODES(height) ( (0x1 << (1+(height))) - 1 )

const NPYSpecList* NCSGData::MakeSPECS()
{
    NPYSpecList* sl = new NPYSpecList(); 

    sl->add( (unsigned)SRC_NODES       , new NPYSpec("srcnodes.npy"       , 0, 4, 4, 0, 0, NPYBase::FLOAT , "" , false));
    sl->add( (unsigned)SRC_IDX         , new NPYSpec("srcidx.npy"         , 0, 4, 0, 0, 0, NPYBase::UINT  , "" , false));
    sl->add( (unsigned)SRC_TRANSFORMS  , new NPYSpec("srctransforms.npy"  , 0, 4, 4, 0, 0, NPYBase::FLOAT , "" , false));
    sl->add( (unsigned)SRC_PLANES      , new NPYSpec("srcplanes.npy"      , 0, 4, 0, 0, 0, NPYBase::FLOAT , "" , true ));
    sl->add( (unsigned)SRC_FACES       , new NPYSpec("srcfaces.npy"       , 0, 3, 0, 0, 0, NPYBase::INT   , "" , true ));
    sl->add( (unsigned)SRC_VERTS       , new NPYSpec("srcverts.npy"       , 0, 3, 0, 0, 0, NPYBase::FLOAT , "" , true ));
    sl->add( (unsigned)NODES           , new NPYSpec("nodes.npy"          , 0, 4, 4, 0, 0, NPYBase::FLOAT , "" , false));
    sl->add( (unsigned)TRANSFORMS      , new NPYSpec("transforms.npy"     , 0, 4, 4, 0, 0, NPYBase::FLOAT , "" , false));
    sl->add( (unsigned)GTRANSFORMS     , new NPYSpec("gtransforms.npy"    , 0, 3, 4, 4, 0, NPYBase::FLOAT , "" , false));
    sl->add( (unsigned)IDX             , new NPYSpec("idx.npy"            , 0, 4, 0, 0, 0, NPYBase::UINT  , "" , false));

    return sl ; 
}
const NPYSpecList* NCSGData::SPECS = MakeSPECS() ; 

NPY<float>*    NCSGData::getNodeBuffer() const {         return dynamic_cast<NPY<float>*>(m_npy->getBuffer((int)NODES)) ; }
NPY<float>*    NCSGData::getTransformBuffer() const {    return dynamic_cast<NPY<float>*>(m_npy->getBuffer((int)TRANSFORMS)) ; }
NPY<float>*    NCSGData::getGTransformBuffer() const {   return dynamic_cast<NPY<float>*>(m_npy->getBuffer((int)GTRANSFORMS)) ; }
NPY<unsigned>* NCSGData::getIdxBuffer() const {          return dynamic_cast<NPY<unsigned>*>(m_npy->getBuffer((int)IDX)) ; }
NPY<float>*    NCSGData::getSrcTransformBuffer() const { return dynamic_cast<NPY<float>*>(m_npy->getBuffer((int)SRC_TRANSFORMS)) ; }
NPY<float>*    NCSGData::getSrcNodeBuffer() const {      return dynamic_cast<NPY<float>*>(m_npy->getBuffer((int)SRC_NODES)) ; } 
NPY<float>*    NCSGData::getSrcPlaneBuffer() const {     return dynamic_cast<NPY<float>*>(m_npy->getBuffer((int)SRC_PLANES)) ; }
NPY<float>*    NCSGData::getSrcVertsBuffer() const {     return dynamic_cast<NPY<float>*>(m_npy->getBuffer((int)SRC_VERTS)) ; }
NPY<int>*      NCSGData::getSrcFacesBuffer() const {     return dynamic_cast<NPY<int>*>(m_npy->getBuffer((int)SRC_FACES)) ; }
NPY<unsigned>* NCSGData::getSrcIdxBuffer() const {       return dynamic_cast<NPY<unsigned>*>(m_npy->getBuffer((int)SRC_IDX)) ; } 


// invoked from NCSG::NCSG(nnode* root )
void NCSGData::init_buffers(unsigned height)
{
   m_height = height ; 
 
   unsigned num_nodes = NumNodes(height); // number of nodes for a complete binary tree of the needed height, with no balancing 
   m_num_nodes = num_nodes ; 

   bool zero = true ; 
   m_npy->initBuffer( (int)NODES         ,  m_num_nodes, zero ); 
   m_npy->initBuffer( (int)TRANSFORMS    ,            0, zero ); 
   m_npy->initBuffer( (int)GTRANSFORMS   ,            0, zero ); 
   m_npy->initBuffer( (int)SRC_NODES     ,  m_num_nodes, zero ); 
   m_npy->initBuffer( (int)SRC_TRANSFORMS,            0, zero ); 
   m_npy->initBuffer( (int)SRC_PLANES    ,            0, zero ); 
   m_npy->initBuffer( (int)SRC_IDX       ,            1, zero ); 
}

// invoked from NCSG::NCSG(const char* treedir)
void NCSGData::loadsrc(const char* treedir)
{
    m_npy->loadBuffer( treedir,(int)SRC_NODES ); 
    m_npy->loadBuffer( treedir,(int)SRC_TRANSFORMS ) ; 
    m_npy->loadBuffer( treedir,(int)SRC_PLANES ); 
    m_npy->loadBuffer( treedir,(int)SRC_VERTS ); 
    m_npy->loadBuffer( treedir,(int)SRC_FACES ); 
    m_npy->loadBuffer( treedir,(int)SRC_IDX); 

    m_num_nodes = m_npy->getNumItems((int)SRC_NODES); 
    m_height = CompleteTreeHeight( m_num_nodes ) ; 

    LOG(info) << " loadsrc DONE " << smry() ; 
}

void NCSGData::savesrc(const char* treedir ) const 
{
    m_npy->saveBuffer( treedir,(int)SRC_NODES ); 
    m_npy->saveBuffer( treedir,(int)SRC_TRANSFORMS ); 
    m_npy->saveBuffer( treedir,(int)SRC_PLANES ); 

    m_npy->saveBuffer( treedir,(int)SRC_VERTS ); 
    m_npy->saveBuffer( treedir,(int)SRC_FACES ); 
    m_npy->saveBuffer( treedir,(int)SRC_IDX ); 
}

NCSGData::NCSGData()  
   :
   m_verbosity(1),
   m_npy(new NPYList(NCSGData::SPECS)),
   m_height(0),
   m_num_nodes(0)
{
}

// must match opticks/analytic/csg.py 

bool NCSGData::Exists(const char* treedir)
{
    return ExistsDir(treedir);
}
bool NCSGData::ExistsDir(const char* treedir)
{
    if(!treedir) return false ; 
    if(!BFile::ExistsDir(treedir)) return false ; 
    return true ; 
}

const char* NCSGData::FILENAME  = "csg.txt" ; 

std::string NCSGData::TxtPath(const char* treedir)
{
    std::string txtpath = BFile::FormPath(treedir, FILENAME) ;
    return txtpath ; 
}
bool NCSGData::ExistsTxt(const char* treedir)
{
    if(!ExistsDir(treedir)) return false ;  
    std::string txtpath = TxtPath(treedir) ; 
    bool exists = BFile::ExistsFile(txtpath.c_str() ); 
    return exists ; 
}


unsigned NCSGData::CompleteTreeHeight( unsigned num_nodes )
{
    unsigned height = UINT_MAX ;  
    int h = MAX_HEIGHT*2 ;   // <-- dont let exceeding MAXHEIGHT, mess up determination of height 
    while(h--)
    {
        unsigned complete_nodes = TREE_NODES(h) ;
        if(complete_nodes == num_nodes) height = h ; 
    }

    bool invalid_height = height == UINT_MAX ; 

    if(invalid_height)
    {
        LOG(fatal) << "NCSGData::CompleteTreeHeight"
                   << " INVALID_HEIGHT "
                   << " num_nodes " << num_nodes
                   << " MAX_HEIGHT " << MAX_HEIGHT
                   ;
    }
    assert(!invalid_height); // must be complete binary tree sized 1, 3, 7, 15, 31, ...
    return height ; 
}

unsigned NCSGData::NumNodes(unsigned height)
{
   return TREE_NODES(height);
}
unsigned NCSGData::getHeight() const 
{
    return m_height ; 
}
unsigned NCSGData::getNumNodes() const 
{
    return m_num_nodes ; 
}


void NCSGData::setIdx( unsigned index, unsigned soIdx, unsigned lvIdx, unsigned height )
{
    assert( height == m_height ); 
    glm::uvec4 uidx(index, soIdx, lvIdx, height); 

    NPY<unsigned>* _srcidx = getSrcIdxBuffer() ;  
    _srcidx->setQuad(uidx, 0u );     
}

std::string NCSGData::smry() const 
{
    std::stringstream ss ; 
    ss 
       << " ht " << std::setw(2) << m_height 
       << " nn " << std::setw(4) << m_num_nodes
       << m_npy->desc()
       ;
    return ss.str();
}

std::string NCSGData::desc() const
{
    std::stringstream ss ; 
    ss << "NCSGData " 
       << " node_sh " << ( m_npy->getBufferShape((int)NODES) )
       << " tran_sh " << ( m_npy->getBufferShape((int)TRANSFORMS) )
       ;
    return ss.str();  
}



// pure access to buffer content is fine here, 
// but buffer manipulations belong up in NCSG 

unsigned NCSGData::getTypeCode(unsigned idx) const 
{
    NPY<float>* srcnodes = getSrcNodeBuffer(); 
    return srcnodes->getUInt(idx,TYPECODE_J,TYPECODE_K,0u);
}
unsigned NCSGData::getTransformIndex(unsigned idx) const 
{
    NPY<float>* srcnodes = getSrcNodeBuffer(); 
    unsigned raw = srcnodes->getUInt(idx,TRANSFORM_J,TRANSFORM_K,0u);
    return raw & SSys::OTHERBIT32 ;   // <-- strip the sign bit  
}
bool NCSGData::isComplement(unsigned idx) const 
{
    NPY<float>* srcnodes = getSrcNodeBuffer(); 
    unsigned raw = srcnodes->getUInt(idx,TRANSFORM_J,TRANSFORM_K,0u);
    return raw & SSys::SIGNBIT32 ;   // pick the sign bit 
}
nquad NCSGData::getQuad(unsigned idx, unsigned j) const 
{
    NPY<float>* srcnodes = getSrcNodeBuffer(); 
    nquad qj ; 
    qj.f = srcnodes->getVQuad(idx, j) ;
    return qj ;
}

void NCSGData::getSrcPlanes(std::vector<glm::vec4>& _planes, unsigned idx, unsigned num_plane ) const 
{
    NPY<float>* srcplanes = getSrcPlaneBuffer();
    unsigned buf_planes = srcplanes->getShape(0); 

    assert( idx < buf_planes );
    assert( idx + num_plane - 1 < buf_planes );

    assert( srcplanes->hasShape(-1,4) );

    for(unsigned i=idx ; i < idx + num_plane ; i++)
    {
        glm::vec4 plane = srcplanes->getQuad(i) ;

        _planes.push_back(plane);    

        if(m_verbosity > 3)
        std::cout << " plane " << std::setw(3) << i 
                  << gpresent(plane)
                  << std::endl ; 

    }
}

void NCSGData::dump_gtransforms() const 
{
    NPY<float>* gtransforms = getGTransformBuffer();
    unsigned ni =  gtransforms ? gtransforms->getNumItems() : 0  ;
    for(unsigned i=0 ; i < ni ; i++)
    {
        const nmat4triple* u_gtran = gtransforms->getMat4TriplePtr(i);
        std::cout 
                  << "[" << std::setw(2) << i << "] " 
                  << *u_gtran 
                  << std::endl ; 
    }
}






///  the below methods should propably be up in NCSG for better clarity 

void NCSGData::prepareForImport()
{
    assert(NTRAN == 3);
    NPY<float>* src = getSrcTransformBuffer(); 
    assert( src && "src buffers are required to prepareForImport "); 

    NPY<float>* transforms = NPY<float>::make_triple_transforms(src) ;
    unsigned ni = src->getNumItems();  
    assert(transforms->hasShape(ni,NTRAN,4,4));
    m_npy->setBuffer( (int)TRANSFORMS, transforms  ); 

    m_npy->initBuffer( (int)GTRANSFORMS, 0, false );
}

void NCSGData::prepareForExport()
{
    m_npy->initBuffer( (int)NODES, m_num_nodes, true );
}

nmat4triple* NCSGData::import_transform_triple(unsigned itra)
{
    assert(NTRAN == 3);
    NPY<float>* transforms = getTransformBuffer(); 
    if(itra == 0 || transforms == NULL ) return NULL ; 

    unsigned num_transforms = transforms->getShape(0); 
    unsigned idx = itra - 1 ; // itra is a 1-based index, with 0 meaning None

    assert( idx < num_transforms );
    assert( transforms->hasShape(-1,NTRAN,4,4) );

    nmat4triple* triple = transforms->getMat4TriplePtr(idx);

    return triple ; 
}

// this is invoked from NCSG::import (constructing node tree from the buffers) 
unsigned NCSGData::addUniqueTransform( const nmat4triple* gtransform )
{
    NPY<float>* gtmp = NPY<float>::make(1,NTRAN,4,4);
    gtmp->zero();
    gtmp->setMat4Triple( gtransform, 0);

    NPY<float>* gtransforms = getGTransformBuffer();
    unsigned gtransform_idx = 1 + gtransforms->addItemUnique( gtmp, 0 ) ; 
    delete gtmp ; 

    return gtransform_idx ; 
}


