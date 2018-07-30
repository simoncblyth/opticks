#include "SSys.hh"
#include "PLOG.hh"

#include "BFile.hh"
#include "BStr.hh"

#include "NPart.h"
#include "NPYBase.hpp"
#include "NPY.hpp"
#include "NParameters.hpp"
#include "NGLMExt.hpp"
#include "NCSGData.hpp"
#include "GLMFormat.hpp"

#define TREE_NODES(height) ( (0x1 << (1+(height))) - 1 )

const char* NCSGData::SRC_NODES_      = "srcnodes.npy" ;      
const char* NCSGData::SRC_IDX_        = "srcidx.npy" ; 
const char* NCSGData::SRC_TRANSFORMS_ = "srctransforms.npy" ; 
const char* NCSGData::SRC_PLANES_     = "srcplanes.npy" ;      
const char* NCSGData::SRC_FACES_      = "srcfaces.npy" ;       
const char* NCSGData::SRC_VERTS_      = "srcverts.npy" ;       

//  results of export 
const char* NCSGData::NODES_          = "nodes.npy" ;         
const char* NCSGData::TRANSFORMS_     = "transforms.npy" ;         
const char* NCSGData::GTRANSFORMS_    = "gtransforms.npy" ;         
const char* NCSGData::IDX_            = "idx.npy" ;

NPYBase::Type_t NCSGData::BufferType(NCSGData_t bid)
{
    NPYBase::Type_t type = NPYBase::FLOAT ; 
    switch(bid)
    {
        case SRC_NODES:        type = NPYBase::FLOAT  ; break ; 
        case SRC_IDX:          type = NPYBase::UINT   ; break ; 
        case SRC_TRANSFORMS:   type = NPYBase::FLOAT  ; break ; 
        case SRC_PLANES:       type = NPYBase::FLOAT  ; break ; 
        case SRC_FACES:        type = NPYBase::INT    ; break ; 
        case SRC_VERTS:        type = NPYBase::FLOAT  ; break ; 
        case NODES:            type = NPYBase::FLOAT  ; break ; 
        case TRANSFORMS:       type = NPYBase::FLOAT  ; break ; 
        case GTRANSFORMS:      type = NPYBase::FLOAT  ; break ; 
        case IDX:              type = NPYBase::UINT   ; break ; 
    }
    return type ; 
}

const char* NCSGData::BufferName(NCSGData_t bid)
{
    const char* name = NULL ; 
    switch(bid)
    {
        case SRC_NODES:        name = SRC_NODES_      ; break ; 
        case SRC_IDX:          name = SRC_IDX_        ; break ; 
        case SRC_TRANSFORMS:   name = SRC_TRANSFORMS_ ; break ; 
        case SRC_PLANES:       name = SRC_PLANES_     ; break ; 
        case SRC_FACES:        name = SRC_FACES_      ; break ; 
        case SRC_VERTS:        name = SRC_VERTS_      ; break ; 
        case NODES:            name = NODES_          ; break ; 
        case TRANSFORMS:       name = TRANSFORMS_     ; break ; 
        case GTRANSFORMS:      name = GTRANSFORMS_    ; break ; 
        case IDX:              name = IDX_            ; break ; 
    }
    return name ; 
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
   //m_num_transforms(0),
   //m_num_planes(0),

   m_meta(new NParameters),

   m_nodes(NULL),
   m_transforms(NULL),
   m_gtransforms(NULL),
   m_planes(NULL),
   m_idx(NULL)
{
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



NPY<float>* NCSGData::getNodeBuffer() const 
{
    return dynamic_cast<NPY<float>*>(m_nodes) ; 
}
NPY<float>* NCSGData::getTransformBuffer() const 
{
    return dynamic_cast<NPY<float>*>(m_transforms) ; 
}
NPY<float>* NCSGData::getGTransformBuffer() const
{
    return dynamic_cast<NPY<float>*>(m_gtransforms) ; 
}


NPY<unsigned>* NCSGData::getIdxBuffer() const 
{
    return dynamic_cast<NPY<unsigned>*>(m_idx) ; 
}


NPY<float>* NCSGData::getSrcTransformBuffer() const 
{
    return dynamic_cast<NPY<float>*>(m_srctransforms) ; 
}
NPY<float>* NCSGData::getSrcNodeBuffer() const 
{
    return dynamic_cast<NPY<float>*>(m_srcnodes) ; 
}
NPY<float>* NCSGData::getSrcPlaneBuffer() const 
{
    return dynamic_cast<NPY<float>*>(m_srcplanes) ; 
}
NPY<float>* NCSGData::getSrcVertsBuffer() const 
{
    return dynamic_cast<NPY<float>*>(m_srcverts) ; 
}
NPY<int>* NCSGData::getSrcFacesBuffer() const 
{
    return dynamic_cast<NPY<int>*>(m_srcfaces) ; 
}
NPY<unsigned>* NCSGData::getSrcIdxBuffer() const 
{
    return dynamic_cast<NPY<unsigned>*>(m_srcidx) ; 
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
    std::stringstream ss ; 
    ss 
       << " ht " << std::setw(2) << m_height 
       << " nn " << std::setw(4) << m_num_nodes
       << " snd " << ( m_srcnodes ? m_srcnodes->getShapeString() : "NULL" )
       << " nd " << ( m_nodes ? m_nodes->getShapeString() : "NULL" )
       << " str " << ( m_srctransforms ? m_srctransforms->getShapeString() : "NULL" )
       << " tr " << ( m_transforms ? m_transforms->getShapeString() : "NULL" )
       << " gtr " << ( m_gtransforms ? m_gtransforms->getShapeString() : "NULL" )
       << " pln " << ( m_srcplanes ? m_srcplanes->getShapeString() : "NULL" )
       ;

    return ss.str();
}


std::string NCSGData::desc() const
{
    std::string node_sh = m_nodes ? m_nodes->getShapeString() : "-" ;    
    std::string tran_sh = m_transforms ? m_transforms->getShapeString() : "-" ;    
    std::stringstream ss ; 
    ss << "NCSGData " 
       << " node_sh " << node_sh  
       << " tran_sh " << tran_sh  
       << " meta " << m_meta->desc()
       ;
    return ss.str();  
}


void NCSGData::savesrc(const char* treedir ) const 
{
    saveSrcNodes(treedir); 
    saveSrcTransforms(treedir);
    saveSrcPlanes(treedir);   
    saveSrcVerts(treedir);   
    saveSrcFaces(treedir);   
    saveSrcIdx(treedir);   

    saveMetadata(treedir, -1 ); 
    saveNodeMetadata(treedir);
}

void NCSGData::loadsrc(const char* treedir)
{
    loadSrcNodes(treedir);        // m_srcnodes,      m_num_nodes,        m_height 
    loadSrcTransforms(treedir);   // m_srctransforms
    loadSrcPlanes(treedir);       // m_srcplanes,     
    loadSrcVerts(treedir);        // m_srcverts     
    loadSrcFaces(treedir);        // m_srcfaces
    loadSrcIdx(treedir);          // m_srcidx

    loadMetadata(treedir);        // m_meta
    loadNodeMetadata(treedir);    // m_modemeta   requires m_num_nodes

    LOG(info) << " loadsrc DONE " << smry() ; 
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

/*
    if(missmeta.size() > 0)
    {
        std::cerr << " missmeta 1-based (" << missmeta.size() << ") : " ; 
        for(unsigned i=0 ; i < missmeta.size() ; i++) std::cerr << " " << missmeta[i]  ; 
        std::cerr << std::endl  ; 
    }
*/

}



void NCSGData::saveSrcNodes(const char* treedir) const
{
/*
    std::string path = BFile::FormPath(treedir, SRC_NODES_ ) ;
    m_srcnodes->save(path.c_str());
*/
    saveBuffer( treedir, SRC_NODES ); 
} 

void NCSGData::loadSrcNodes(const char* treedir)
{
    loadBuffer( treedir, SRC_NODES ); 

/*
    std::string path = BFile::FormPath(treedir, SRC_NODES_ ) ;
    m_srcnodes = NPY<float>::load(path.c_str());

    if(!m_srcnodes)
    {
         LOG(fatal) << "NCSGData::loadSrcNodes"
                    << " failed to load " 
                    << " path [" << path << "]"
                    ;
    }
*/

    NPY<float>* srcnodes = getSrcNodeBuffer(); 
    assert(srcnodes);
 
    // hmm dont like doing processing on load that 
    // as wil not be done when getting from node tree ?

    m_num_nodes  = srcnodes->getShape(0) ;  
    unsigned nj = srcnodes->getShape(1);
    unsigned nk = srcnodes->getShape(2);
    assert( nj == NJ );
    assert( nk == NK );
    m_height = CompleteTreeHeight( m_num_nodes ) ; 
}


void NCSGData::saveSrcTransforms(const char* treedir) const
{
    saveBuffer( treedir, SRC_TRANSFORMS ); 

/*
    std::string path = BFile::FormPath(treedir, SRC_TRANSFORMS_) ;
    m_srctransforms->save(path.c_str());
*/
} 

void NCSGData::loadSrcTransforms(const char* treedir)
{
    loadBuffer( treedir, SRC_TRANSFORMS ) ; 
    NPY<float>* a = getSrcTransformBuffer();
    if(a)
    {
        assert( a->hasShape(-1,4,4) );
    }

/*
    std::string path = BFile::FormPath(treedir, SRC_TRANSFORMS_) ;
    if(!BFile::ExistsFile(path.c_str())) return ; 

    m_srctransforms = NPY<float>::load(path.c_str());
    NPY<float>* src = m_srctransforms ; 
    assert(src); 
    bool valid_src = src->hasShape(-1,4,4) ;
    if(!valid_src) 
    {
        LOG(fatal) << "NCSGData::loadSrcTransforms"
                   << " invalid src transforms "
                   << " path " << path
                   << " srctransforms " << src->getShapeString()
                   ;
    }

    NPY<float>* srctransforms = getSrcTransformBuffer();

    bool valid_src = srctransforms->hasShape(-1,4,4) ;
    assert(valid_src);
*/

    //unsigned ni = srctransforms->getShape(0) ;
    //m_num_transforms  = ni  ;  
}



void NCSGData::saveSrcVerts(const char* treedir) const
{
    saveBuffer( treedir, SRC_VERTS ); 

/*
    std::string path = BFile::FormPath(treedir, SRC_VERTS_) ;
    if(!m_srcverts) return ; 
    m_srcverts->save(path.c_str());
*/
} 

void NCSGData::loadSrcVerts(const char* treedir)
{
    loadBuffer( treedir, SRC_VERTS ); 

    NPY<float>* a = getSrcVertsBuffer(); 
    assert(NPYBase::HasShape(a, -1, 3)) ; 

/*
    std::string path = BFile::FormPath(treedir,  SRC_VERTS_ ) ;
    if(!BFile::ExistsFile(path.c_str())) return ; 

    NPY<float>* a = NPY<float>::load(path.c_str());
    assert(a); 
    bool valid = a->hasShape(-1,3) ;
    if(!valid) 
    {
        LOG(fatal) << "NCSGData::loadVerts"
                   << " invalid verts  "
                   << " path " << path
                   << " shape " << a->getShapeString()
                   ;
    }

    assert(valid);
    m_srcverts = a ; 
    m_num_srcverts = a->getShape(0) ; 

    LOG(info) << " loaded " << m_num_srcverts ;
*/

}



void NCSGData::saveSrcFaces(const char* treedir) const
{
    saveBuffer( treedir, SRC_FACES ); 
/*
    std::string path = BFile::FormPath(treedir, SRC_FACES_) ;
    if(!m_srcfaces) return ; 
    m_srcfaces->save(path.c_str());
*/
} 

void NCSGData::loadSrcFaces(const char* treedir)
{
    loadBuffer( treedir, SRC_FACES ); 

    NPYBase* buf = getBuffer(SRC_FACES) ; 
    if(buf)
    {
        assert( buf->hasShape(-1,4) ); 
    }

/*
    std::string path = BFile::FormPath(treedir,  SRC_FACES_ ) ;
    if(!BFile::ExistsFile(path.c_str())) return ; 

    NPY<int>* a = NPY<int>::load(path.c_str());
    assert(a); 
    bool valid = a->hasShape(-1,4) ;
    if(!valid) 
    {
        LOG(fatal) << "NCSGData::loadFaces"
                   << " invalid faces  "
                   << " path " << path
                   << " shape " << a->getShapeString()
                   ;
    }
    assert(valid);
    m_srcfaces = a ; 
    m_num_srcfaces = a->getShape(0) ; 

    LOG(info) << " loaded " << m_num_srcfaces ;

*/

}

void NCSGData::saveSrcPlanes(const char* treedir) const
{
    saveBuffer( treedir, SRC_PLANES ); 
/*
    std::string path = BFile::FormPath(treedir, SRC_PLANES) ;
    if(!m_srcplanes) return ; 

    if(m_srcplanes->getNumItems() > 0)
        m_srcplanes->save(path.c_str()); 
*/
} 

void NCSGData::loadSrcPlanes(const char* treedir)
{
    loadBuffer( treedir, SRC_PLANES ); 
    NPYBase* buf = getBuffer(SRC_PLANES); 
    if(buf) 
    {
        assert( buf->hasShape(-1,4) );
    }

/*
    std::string path = BFile::FormPath(treedir,  SRC_PLANES ) ;
    if(!BFile::ExistsFile(path.c_str())) return ; 

    NPY<float>* a = NPY<float>::load(path.c_str());
    assert(a); 
    bool valid = a->hasShape(-1,4) ;
    if(!valid) 
    {
        LOG(fatal) << "NCSGData::loadSrcPlanes"
                   << " invalid planes  "
                   << " path " << path
                   << " shape " << a->getShapeString()
                   ;
    }
    assert(valid);

    m_srcplanes = a ; 
    m_num_planes = a->getShape(0) ; 
*/

}


void NCSGData::saveSrcIdx(const char* treedir) const
{
    saveBuffer( treedir, SRC_IDX ); 

/*
    std::string path = BFile::FormPath(treedir, SRC_IDX) ;
    if(!m_srcidx) return ; 
    m_srcidx->save(path.c_str());
*/
} 

void NCSGData::loadSrcIdx(const char* treedir)
{
    loadBuffer(treedir, SRC_IDX); 
    NPYBase* buf = getBuffer(SRC_IDX); 
    assert( buf->hasShape(1,4) ); 

/*
    std::string path = BFile::FormPath(treedir, SRC_IDX ) ;
    //if(!BFile::ExistsFile(path.c_str())) return ; 

    NPY<unsigned>* a = NPY<unsigned>::load(path.c_str());
    bool valid = a->hasShape(1,4) ;
    if(!valid) 
    {
        LOG(fatal) << "NCSGData::loadSrcIdx"
                   << " invalid idx  "
                   << " path " << path
                   << " shape " << a->getShapeString()
                   ;
    }
    assert(valid);
    m_srcidx = a ;
*/
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


/*
    m_planes = NPY<float>::make(0,4);
    m_planes->zero();
*/
}



nmat4pair* NCSGData::import_transform_pair(unsigned itra)
{
    assert(NTRAN == 2);
    NPY<float>* transforms = getTransformBuffer(); 
    if(itra == 0 || transforms == NULL) return NULL ; // itra is a 1-based index, with 0 meaning None

    unsigned num_transforms = transforms->getShape(0); 
    unsigned idx = itra - 1 ;    // convert to 0-based idx 

    assert( idx < num_transforms );
    assert( transforms->hasShape(-1,NTRAN,4,4) );

    nmat4pair* pair = transforms->getMat4PairPtr(idx);

    return pair ; 
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

