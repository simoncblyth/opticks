#include "SSys.hh"
#include "PLOG.hh"

#include "BFile.hh"
#include "BStr.hh"

#include "NPart.h"

#include "NPYSpec.hpp"
#include "NPYSpecList.hpp"
#include "NPYBase.hpp"
#include "NPY.hpp"
#include "NParameters.hpp"
#include "NGLMExt.hpp"
#include "NCSGData.hpp"
#include "GLMFormat.hpp"

#define TREE_NODES(height) ( (0x1 << (1+(height))) - 1 )


const NPYSpecList* NCSGData::MakeSPECS()
{
    NPYSpecList* sl = new NPYSpecList(); 

    sl->add( (unsigned)SRC_NODES       , new NPYSpec("srcnodes.npy"       , 0, 4, 4, 0, 0, NPYBase::FLOAT , "" ) );
    sl->add( (unsigned)SRC_IDX         , new NPYSpec("srcidx.npy"         , 0, 4, 0, 0, 0, NPYBase::UINT  , "" ) );
    sl->add( (unsigned)SRC_TRANSFORMS  , new NPYSpec("srctransforms.npy"  , 0, 4, 4, 0, 0, NPYBase::FLOAT , "" ) );
    sl->add( (unsigned)SRC_PLANES      , new NPYSpec("srcplanes.npy"      , 0, 4, 0, 0, 0, NPYBase::FLOAT , "" ) );
    sl->add( (unsigned)SRC_FACES       , new NPYSpec("srcfaces.npy"       , 0, 3, 0, 0, 0, NPYBase::INT   , "" ) );
    sl->add( (unsigned)SRC_VERTS       , new NPYSpec("srcverts.npy"       , 0, 3, 0, 0, 0, NPYBase::FLOAT , "" ) );
    sl->add( (unsigned)NODES           , new NPYSpec("nodes.npy"          , 0, 4, 4, 0, 0, NPYBase::FLOAT , "" ) );
    sl->add( (unsigned)TRANSFORMS      , new NPYSpec("transforms.npy"     , 0, 4, 4, 0, 0, NPYBase::FLOAT , "" ) );
    sl->add( (unsigned)GTRANSFORMS     , new NPYSpec("gtransforms.npy"    , 0, 4, 4, 0, 0, NPYBase::FLOAT , "" ) );
    sl->add( (unsigned)IDX             , new NPYSpec("idx.npy"            , 0, 4, 0, 0, 0, NPYBase::UINT  , "" ) );

    return sl ; 
}

const NPYSpecList* NCSGData::SPECS = MakeSPECS() ; 

// TODO: try a NPYBaseList to hold the buffers, rather than the member vars 
//       for generic buffer mechanics : and to eliminate case methods like the below

void NCSGData::setBuffer( NCSGData_t bid, NPYBase* buffer )
{
    switch(bid)
    {
        case SRC_NODES:        m_srcnodes = buffer      ; break ; 
        case SRC_IDX:          m_srcidx   = buffer      ; break ; 
        case SRC_TRANSFORMS:   m_srctransforms = buffer ; break ; 
        case SRC_PLANES:       m_srcplanes = buffer     ; break ; 
        case SRC_FACES:        m_srcfaces = buffer      ; break ; 
        case SRC_VERTS:        m_srcverts = buffer      ; break ; 
        case NODES:            m_nodes = buffer         ; break ; 
        case TRANSFORMS:       m_transforms = buffer    ; break ; 
        case GTRANSFORMS:      m_gtransforms = buffer   ; break ; 
        case IDX:              m_idx = buffer           ; break ; 
    }
}
NPYBase* NCSGData::getBuffer(NCSGData_t bid) const 
{
    NPYBase* buffer = NULL ; 
    switch(bid)
    {
        case SRC_NODES:        buffer = m_srcnodes      ; break ; 
        case SRC_IDX:          buffer = m_srcidx        ; break ; 
        case SRC_TRANSFORMS:   buffer = m_srctransforms ; break ; 
        case SRC_PLANES:       buffer = m_srcplanes     ; break ; 
        case SRC_FACES:        buffer = m_srcfaces      ; break ; 
        case SRC_VERTS:        buffer = m_srcverts      ; break ; 
        case NODES:            buffer = m_nodes         ; break ; 
        case TRANSFORMS:       buffer = m_transforms    ; break ; 
        case GTRANSFORMS:      buffer = m_gtransforms   ; break ; 
        case IDX:              buffer = m_idx           ; break ; 
    }
    return buffer ; 
}

NPY<float>*    NCSGData::getNodeBuffer() const {         return dynamic_cast<NPY<float>*>(getBuffer(NODES)) ; }
NPY<float>*    NCSGData::getTransformBuffer() const {    return dynamic_cast<NPY<float>*>(getBuffer(TRANSFORMS)) ; }
NPY<float>*    NCSGData::getGTransformBuffer() const {   return dynamic_cast<NPY<float>*>(getBuffer(GTRANSFORMS)) ; }
NPY<unsigned>* NCSGData::getIdxBuffer() const {          return dynamic_cast<NPY<unsigned>*>(getBuffer(IDX)) ; }
NPY<float>*    NCSGData::getSrcTransformBuffer() const { return dynamic_cast<NPY<float>*>(getBuffer(SRC_TRANSFORMS)) ; }
NPY<float>*    NCSGData::getSrcNodeBuffer() const {      return dynamic_cast<NPY<float>*>(getBuffer(SRC_NODES)) ; } 
NPY<float>*    NCSGData::getSrcPlaneBuffer() const {     return dynamic_cast<NPY<float>*>(getBuffer(SRC_PLANES)) ; }
NPY<float>*    NCSGData::getSrcVertsBuffer() const {     return dynamic_cast<NPY<float>*>(getBuffer(SRC_VERTS)) ; }
NPY<int>*      NCSGData::getSrcFacesBuffer() const {     return dynamic_cast<NPY<int>*>(getBuffer(SRC_FACES)) ; }
NPY<unsigned>* NCSGData::getSrcIdxBuffer() const {       return dynamic_cast<NPY<unsigned>*>(getBuffer(SRC_IDX)) ; } 


NCSGData::NCSGData()  
   :
   m_verbosity(1),

   m_srcnodes(NULL),
   m_srctransforms(NULL),
   m_srcplanes(NULL),
   m_srcverts(NULL),
   m_srcfaces(NULL),
   m_srcidx(NULL),

   m_height(0),
   m_num_nodes(0),

   m_meta(new NParameters),

   m_nodes(NULL),
   m_transforms(NULL),
   m_gtransforms(NULL),
   m_planes(NULL),
   m_idx(NULL)
{
}




// TODO: 
//    as intended the below block of methods are generic and can work for any set of buffers 
//    ... so reposition them elsewhere  (perhaps NPYSpecList + NPYBaseList) for reuse from elsewhere


const NPYSpec* NCSGData::BufferSpec( NCSGData_t bid)
{
    return SPECS->getByIdx( (unsigned)bid );  
}
NPYBase::Type_t NCSGData::BufferType(NCSGData_t bid)
{
    return BufferSpec(bid)->getType() ; 
}
const char* NCSGData::BufferName(NCSGData_t bid)
{
    return BufferSpec(bid)->getName() ; 
}

std::string NCSGData::BufferPath( const char* treedir, NCSGData_t bid ) // static
{
    std::string path = BFile::FormPath(treedir, BufferName(bid) ) ;
    return path ; 
}

void NCSGData::saveBuffer(const char* treedir, NCSGData_t bid, bool require) const
{
    NPYBase* buffer = getBuffer(bid); 
    if( require && buffer == NULL )
    {
        LOG(fatal) << " required buffer is NULL  " << BufferName(bid) ; 
        assert(0) ; 
    }
    if( buffer == NULL ) return ; 
    std::string path = BufferPath(treedir, bid); 
    buffer->save(path.c_str());  
}

void NCSGData::loadBuffer(const char* treedir, NCSGData_t bid, bool require)
{
    const NPYSpec* spec = BufferSpec(bid) ; 
    std::string path = BufferPath(treedir, bid); 

    bool exists = BFile::ExistsFile(path.c_str()) ;
    if(require && !exists)
    {
        LOG(fatal) << " required buffer does not exists " << path ;
        assert(0) ; 
    }
    if(!exists) return ; 

    NPYBase::Type_t type = BufferType(bid); 
    NPYBase* buffer = NPYBase::Load( path.c_str(), type );     
    assert( buffer->hasItemSpec( spec )); 

    setBuffer(bid, buffer ); 
    NPYBase* buffer2 = getBuffer(bid) ; 
    assert( buffer == buffer2 ); 
}   


// must match opticks/analytic/csg.py 
const char* NCSGData::FILENAME  = "csg.txt" ; 
const char* NCSGData::TREE_META = "meta.json" ;
const char* NCSGData::NODE_META = "nodemeta.json" ;

const unsigned NCSGData::NTRAN = 3 ; 

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

std::string NCSGData::MetaPath(const char* treedir, int idx)
{
    std::string metapath = idx == -1 ? BFile::FormPath(treedir, TREE_META) : BFile::FormPath(treedir, BStr::itoa(idx), NODE_META) ;
    return metapath ; 
}

bool NCSGData::ExistsMeta(const char* treedir, int idx)
{
    std::string metapath = MetaPath(treedir, idx) ;
    return BFile::ExistsFile(metapath.c_str()) ;
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




NParameters* NCSGData::getMetaParameters(int idx) const
{
    return idx < 0 ? m_meta : getNodeMetadata(unsigned(idx)) ; 
}
NParameters* NCSGData::getNodeMetadata(unsigned idx) const 
{
    return m_nodemeta.count(idx) == 1 ? m_nodemeta.at(idx) : NULL ; 
}



// invoked from NCSG::NCSG(nnode* root )
void NCSGData::init_buffers(unsigned height)
{
   m_height = height ; 
 
   unsigned num_nodes = NumNodes(height); // number of nodes for a complete binary tree of the needed height, with no balancing 
   m_num_nodes = num_nodes ; 

    // TODO : use the spec to do the below generically with  NPYBase::Make(ni, itemspec)

   NPY<float>* nodes = NPY<float>::make( m_num_nodes, NJ, NK);
   nodes->zero();
   setBuffer( NODES, nodes ); 

   NPY<float>* transforms = NPY<float>::make(0,NTRAN,4,4) ;  
   transforms->zero();
   setBuffer( TRANSFORMS, transforms );  

   NPY<float>* gtransforms = NPY<float>::make(0,NTRAN,4,4) ; 
   gtransforms->zero();
   setBuffer( GTRANSFORMS, gtransforms );  



   NPY<float>* srcnodes = NPY<float>::make( m_num_nodes, NJ, NK);
   srcnodes->zero();
   setBuffer( SRC_NODES,  srcnodes ); 
 
   NPY<float>* srctransforms = NPY<float>::make(0,4,4) ;  
   srctransforms->zero();
   setBuffer( SRC_TRANSFORMS, srctransforms );

   NPY<float>* srcplanes = NPY<float>::make(0,4);
   srcplanes->zero();
   setBuffer( SRC_PLANES, srcplanes ); 
 
   NPY<unsigned>* srcidx = NPY<unsigned>::make(1,4);
   srcidx->zero();
   setBuffer( SRC_IDX , srcidx ) ; 

}





void NCSGData::setIdx( unsigned index, unsigned soIdx, unsigned lvIdx, unsigned height )
{
    if(m_srcidx == NULL)
    {
        NPY<unsigned>* srcidx = NPY<unsigned>::make(1, 4);
        srcidx->zero() ;  
        setBuffer( SRC_IDX , srcidx ); 
    } 
    assert( height == m_height ); 
    glm::uvec4 uidx(index, soIdx, lvIdx, height); 

    NPY<unsigned>* _srcidx = getSrcIdxBuffer() ;  
    _srcidx->setQuad(uidx, 0u );     

}




std::string NCSGData::smry() const 
{
    // TODO: generic shapes, avoid using m_srcnodes etc..

    std::stringstream ss ; 
    ss 
       << " ht " << std::setw(2) << m_height 
       << " nn " << std::setw(4) << m_num_nodes
       << " snd " << ( m_srcnodes      ? m_srcnodes->getShapeString() : "NULL" )
       << " nd " << (  m_nodes         ? m_nodes->getShapeString() : "NULL" )
       << " str " << ( m_srctransforms ? m_srctransforms->getShapeString() : "NULL" )
       << " tr " << (  m_transforms    ? m_transforms->getShapeString() : "NULL" )
       << " gtr " << ( m_gtransforms   ? m_gtransforms->getShapeString() : "NULL" )
       << " pln " << ( m_srcplanes     ? m_srcplanes->getShapeString() : "NULL" )
       ;

    return ss.str();
}


std::string NCSGData::desc() const
{
    std::stringstream ss ; 
    ss << "NCSGData " 
       << " node_sh " << ( m_nodes ?      m_nodes->getShapeString() : "-" )
       << " tran_sh " << ( m_transforms ? m_transforms->getShapeString() : "-" )
       << " meta " << m_meta->desc()
       ;
    return ss.str();  
}



void NCSGData::loadsrc(const char* treedir)
{
    loadBuffer( treedir, SRC_NODES ); 
    loadBuffer( treedir, SRC_TRANSFORMS ) ; 
    loadBuffer( treedir, SRC_PLANES ); 
    loadBuffer( treedir, SRC_VERTS ); 
    loadBuffer( treedir, SRC_FACES ); 
    loadBuffer( treedir, SRC_IDX); 

    NPY<float>* srcnodes = getSrcNodeBuffer(); 
    m_num_nodes  = srcnodes->getShape(0) ;  
    m_height = CompleteTreeHeight( m_num_nodes ) ; 

    loadMetadata(treedir);        // m_meta
    loadNodeMetadata(treedir);    // m_modemeta   requires m_num_nodes

    LOG(info) << " loadsrc DONE " << smry() ; 
}


void NCSGData::savesrc(const char* treedir ) const 
{
    saveBuffer( treedir, SRC_NODES ); 
    saveBuffer( treedir, SRC_TRANSFORMS ); 
    saveBuffer( treedir, SRC_PLANES ); 

    saveBuffer( treedir, SRC_VERTS ); 
    saveBuffer( treedir, SRC_FACES ); 
    saveBuffer( treedir, SRC_IDX ); 

    saveMetadata(treedir, -1 ); 
    saveNodeMetadata(treedir);
}


unsigned NCSGData::getTypeCode(unsigned idx)
{
    NPY<float>* srcnodes = getSrcNodeBuffer(); 
    return srcnodes->getUInt(idx,TYPECODE_J,TYPECODE_K,0u);
}
unsigned NCSGData::getTransformIndex(unsigned idx)
{
    NPY<float>* srcnodes = getSrcNodeBuffer(); 
    unsigned raw = srcnodes->getUInt(idx,TRANSFORM_J,TRANSFORM_K,0u);
    return raw & SSys::OTHERBIT32 ;   // <-- strip the sign bit  
}
bool NCSGData::isComplement(unsigned idx)
{
    NPY<float>* srcnodes = getSrcNodeBuffer(); 
    unsigned raw = srcnodes->getUInt(idx,TRANSFORM_J,TRANSFORM_K,0u);
    return raw & SSys::SIGNBIT32 ;   // pick the sign bit 
}
nquad NCSGData::getQuad(unsigned idx, unsigned j)
{
    NPY<float>* srcnodes = getSrcNodeBuffer(); 
    nquad qj ; 
    qj.f = srcnodes->getVQuad(idx, j) ;
    return qj ;
}



void NCSGData::prepareForImport()
{
    assert(NTRAN == 2 || NTRAN == 3);
    NPY<float>* src = getSrcTransformBuffer(); 

    assert( src && "src buffers are required to prepareForImport "); 

    NPY<float>* transforms = NTRAN == 2 ? NPY<float>::make_paired_transforms(src) : NPY<float>::make_triple_transforms(src) ;
    unsigned ni = src->getNumItems();  
    assert(transforms->hasShape(ni,NTRAN,4,4));
    setBuffer( TRANSFORMS, transforms  ); 

    NPY<float>* gtransforms = NPY<float>::make(0,NTRAN,4,4) ;  // for collecting unique gtransforms
    setBuffer( GTRANSFORMS, gtransforms );
}

void NCSGData::prepareForExport()
{
    NPY<float>* nodes = NPY<float>::make( m_num_nodes, NJ, NK);
    nodes->zero();

    setBuffer( NODES, nodes ) ; 
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



void NCSGData::saveMetadata(const char* treedir, int idx) const 
{
    NParameters* meta = getMetaParameters(idx) ; 
    if(!meta) return ; 

    std::string metapath = MetaPath(treedir, idx) ;
    meta->save(metapath.c_str()); 
}

void NCSGData::loadMetadata(const char* treedir)
{
    m_meta = LoadMetadata( treedir, -1 );  // -1:treemeta
    if(!m_meta)
    { 
        LOG(fatal) << "NCSGData::loadMetadata"
                    << " treedir " << treedir 
                    ;
    } 
    assert(m_meta);
}

NParameters* NCSGData::LoadMetadata(const char* treedir, int idx )
{
    std::string path = MetaPath(treedir, idx) ;
    NParameters* meta = new NParameters ; 
    if(BFile::ExistsFile(path.c_str()))
    {
         meta->load_( path.c_str() );
    } 
    else
    {
        // lots missing is expected as are looping over complete tree node count
        // see notes/issues/axel_GSceneTest_fail.rst
        LOG(trace) << "NCSGData::LoadMetadata"
                     << " missing metadata "
                     << " treedir " << treedir  
                     << " idx " << idx
                     << " path " << path
                     ;
    }
    return meta ; 
}

void NCSGData::saveNodeMetadata(const char* treedir) const 
{
    for(unsigned idx=0 ; idx < m_num_nodes ; idx++)
    {
        saveMetadata(treedir, idx); 
    } 
}

void NCSGData::loadNodeMetadata(const char* treedir)
{
    // FIX: this idx is not same as real complete binary tree node_idx ?

    // just looping over the number of nodes in the complete tree
    // so it is not surprising that many of them are missing metadata

    std::vector<unsigned> missmeta ;
    assert(m_num_nodes > 0);
    for(unsigned idx=0 ; idx < m_num_nodes ; idx++)
    {
        NParameters* nodemeta = LoadMetadata(treedir, idx);
        if(nodemeta)
        { 
            m_nodemeta[idx] = nodemeta ; 
        }
        else
        {
            missmeta.push_back(idx+1);  // make 1-based 
        }
    } 

    LOG(debug) << "NCSGData::loadNodeMetadata" 
              << " treedir " << treedir
              << " m_height " << m_height
              << " m_num_nodes " << m_num_nodes
              << " missmeta " << missmeta.size()
               ; 
}







// metadata from the root nodes of the CSG trees for each solid
// pmt-cd treebase.py:Node._get_meta
//
template<typename T>
T NCSGData::getMeta(const char* key, const char* fallback ) const 
{
    return m_meta ? m_meta->get<T>(key, fallback) : BStr::LexicalCast<T>(fallback) ;
}

template<typename T>
void NCSGData::setMeta(const char* key, T value )
{
    assert( m_meta ) ; 
    return m_meta->set<T>(key, value) ;
}

template NPY_API void NCSGData::setMeta<float>(const char*, float);
template NPY_API void NCSGData::setMeta<int>(const char*, int);
template NPY_API void NCSGData::setMeta<bool>(const char*, bool);
template NPY_API void NCSGData::setMeta<std::string>(const char*, std::string);

template NPY_API std::string NCSGData::getMeta<std::string>(const char*, const char*) const ;
template NPY_API int         NCSGData::getMeta<int>(const char*, const char*) const ;
template NPY_API float       NCSGData::getMeta<float>(const char*, const char*) const ;
template NPY_API bool        NCSGData::getMeta<bool>(const char*, const char*) const ;

