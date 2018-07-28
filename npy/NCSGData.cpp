#include "SSys.hh"
#include "PLOG.hh"

#include "BFile.hh"
#include "BStr.hh"

#include "NPart.h"
#include "NPY.hpp"
#include "NParameters.hpp"
#include "NGLMExt.hpp"
#include "NCSGData.hpp"
#include "GLMFormat.hpp"

#define TREE_NODES(height) ( (0x1 << (1+(height))) - 1 )

// must match opticks/analytic/csg.py 
const char* NCSGData::FILENAME  = "csg.txt" ; 

const char* NCSGData::TREE_META = "meta.json" ;
const char* NCSGData::NODE_META = "nodemeta.json" ;

const char* NCSGData::SRC_PLANES     = "srcplanes.npy" ; 
const char* NCSGData::SRC_FACES      = "srcfaces.npy" ; 
const char* NCSGData::SRC_VERTS      = "srcverts.npy" ; 
const char* NCSGData::SRC_TRANSFORMS = "srctransforms.npy" ; 
const char* NCSGData::SRC_IDX        = "srcidx.npy" ; 
const char* NCSGData::SRC_NODES      = "srcnodes.npy" ;

//  results of export 
const char* NCSGData::PLANES = "planes.npy" ; 
const char* NCSGData::NODES  = "nodes.npy" ;
const char* NCSGData::IDX    = "idx.npy" ;


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
   m_num_transforms(0),
   m_num_planes(0),
   m_num_srcverts(0),
   m_num_srcfaces(0),

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
    return m_nodes ; 
}
NPY<float>* NCSGData::getTransformBuffer() const 
{
    return m_transforms ; 
}
NPY<float>* NCSGData::getGTransformBuffer() const
{
    return m_gtransforms ; 
}
NPY<float>* NCSGData::getPlaneBuffer() const
{
    return m_planes ; 
}
NPY<unsigned>* NCSGData::getIdxBuffer() const 
{
    return m_idx ; 
}


NPY<float>* NCSGData::getSrcTransformBuffer() const 
{
    return m_srctransforms ; 
}
NPY<float>* NCSGData::getSrcNodeBuffer() const 
{
    return m_srcnodes ; 
}
NPY<float>* NCSGData::getSrcPlaneBuffer() const 
{
    return m_srcplanes ; 
}
NPY<float>* NCSGData::getSrcVertsBuffer() const 
{
    return m_srcverts ; 
}
NPY<int>* NCSGData::getSrcFacesBuffer() const 
{
    return m_srcfaces ; 
}
NPY<unsigned>* NCSGData::getSrcIdxBuffer() const 
{
    return m_srcidx ; 
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

   m_nodes = NPY<float>::make( m_num_nodes, NJ, NK);
   m_nodes->zero();

   m_transforms = NPY<float>::make(0,NTRAN,4,4) ;  
   m_transforms->zero();

   m_gtransforms = NPY<float>::make(0,NTRAN,4,4) ; 
   m_gtransforms->zero();


   m_srcnodes = NPY<float>::make( m_num_nodes, NJ, NK);
   m_srcnodes->zero();

   m_srctransforms = NPY<float>::make(0,4,4) ;  
   m_srctransforms->zero();

   m_srcplanes = NPY<float>::make(0,4);
   m_srcplanes->zero();

   m_srcidx = NPY<unsigned>::make(1,4);
   m_srcidx->zero();
}




void NCSGData::setIdx( unsigned index, unsigned soIdx, unsigned lvIdx, unsigned height )
{
    if(m_srcidx == NULL)
    {
        m_srcidx = NPY<unsigned>::make(1, 4);
        m_srcidx->zero() ;  
    } 
    assert( height == m_height ); 
    glm::uvec4 uidx(index, soIdx, lvIdx, height); 
    m_srcidx->setQuad(uidx, 0u );     
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
    loadSrcNodes(treedir);        // m_nodes, m_num_nodes, m_height 
    loadSrcTransforms(treedir);   // m_transforms, m_num_transforms
    loadSrcPlanes(treedir);       // m_planes, m_num_planes 
    loadSrcVerts(treedir);        // m_srcverts m_num_srcverts
    loadSrcFaces(treedir);        // m_srcfaces, m_num_srcfaces
    loadSrcIdx(treedir);          // m_srcidx

    loadMetadata(treedir);      // m_meta
    loadNodeMetadata(treedir);  // requires m_num_nodes

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
    std::string path = BFile::FormPath(treedir, SRC_NODES) ;
    m_srcnodes->save(path.c_str());
} 

void NCSGData::loadSrcNodes(const char* treedir)
{
    std::string path = BFile::FormPath(treedir, SRC_NODES) ;
    m_srcnodes = NPY<float>::load(path.c_str());

    if(!m_srcnodes)
    {
         LOG(fatal) << "NCSGData::loadSrcNodes"
                    << " failed to load " 
                    << " path [" << path << "]"
                    ;
    }

    assert(m_srcnodes);
 
    // hmm dont like doing processing on load that 
    // as wil not be done when getting from node tree ?

    m_num_nodes  = m_srcnodes->getShape(0) ;  
    unsigned nj = m_srcnodes->getShape(1);
    unsigned nk = m_srcnodes->getShape(2);
    assert( nj == NJ );
    assert( nk == NK );
    m_height = CompleteTreeHeight( m_num_nodes ) ; 
}


void NCSGData::saveSrcTransforms(const char* treedir) const
{
    std::string path = BFile::FormPath(treedir, SRC_TRANSFORMS) ;
    m_srctransforms->save(path.c_str());
    // gtransforms not saved, they are constructed on load
} 

void NCSGData::loadSrcTransforms(const char* treedir)
{
    std::string path = BFile::FormPath(treedir, SRC_TRANSFORMS) ;
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

    assert(valid_src);
    unsigned ni = src->getShape(0) ;
    m_num_transforms  = ni  ;  
}


void NCSGData::prepareForImport()
{
    assert(NTRAN == 2 || NTRAN == 3);
    NPY<float>* src = m_srctransforms ; 

    assert( src && "src buffers are required to prepareForImport "); 

    NPY<float>* transforms = NTRAN == 2 ? NPY<float>::make_paired_transforms(src) : NPY<float>::make_triple_transforms(src) ;
    unsigned ni = src->getNumItems();  
    assert(transforms->hasShape(ni,NTRAN,4,4));

    m_transforms = transforms ; 
    m_gtransforms = NPY<float>::make(0,NTRAN,4,4) ;  // for collecting unique gtransforms
}

void NCSGData::prepareForExport()
{
    m_nodes = NPY<float>::make( m_num_nodes, NJ, NK);
    m_nodes->zero();

    m_planes = NPY<float>::make(0,4);
    m_planes->zero();
}


void NCSGData::saveSrcVerts(const char* treedir) const
{
    std::string path = BFile::FormPath(treedir, SRC_VERTS) ;
    if(!m_srcverts) return ; 
    m_srcverts->save(path.c_str());
} 

void NCSGData::loadSrcVerts(const char* treedir)
{
    std::string path = BFile::FormPath(treedir,  SRC_VERTS ) ;
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
}



void NCSGData::saveSrcFaces(const char* treedir) const
{
    std::string path = BFile::FormPath(treedir, SRC_FACES) ;
    if(!m_srcfaces) return ; 
    m_srcfaces->save(path.c_str());
} 

void NCSGData::loadSrcFaces(const char* treedir)
{
    std::string path = BFile::FormPath(treedir,  SRC_FACES ) ;
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
}

void NCSGData::saveSrcPlanes(const char* treedir) const
{
    std::string path = BFile::FormPath(treedir, SRC_PLANES) ;
    if(!m_srcplanes) return ; 

    if(m_srcplanes->getNumItems() > 0)
        m_srcplanes->save(path.c_str()); 
} 

void NCSGData::loadSrcPlanes(const char* treedir)
{
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
}


void NCSGData::saveSrcIdx(const char* treedir) const
{
    std::string path = BFile::FormPath(treedir, SRC_IDX) ;
    if(!m_srcidx) return ; 
    m_srcidx->save(path.c_str());
} 

void NCSGData::loadSrcIdx(const char* treedir)
{
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
}


unsigned NCSGData::getTypeCode(unsigned idx)
{
    return m_srcnodes->getUInt(idx,TYPECODE_J,TYPECODE_K,0u);
}
unsigned NCSGData::getTransformIndex(unsigned idx)
{
    unsigned raw = m_srcnodes->getUInt(idx,TRANSFORM_J,TRANSFORM_K,0u);
    return raw & SSys::OTHERBIT32 ;   // <-- strip the sign bit  
}
bool NCSGData::isComplement(unsigned idx)
{
    unsigned raw = m_srcnodes->getUInt(idx,TRANSFORM_J,TRANSFORM_K,0u);
    return raw & SSys::SIGNBIT32 ;   // pick the sign bit 
}
nquad NCSGData::getQuad(unsigned idx, unsigned j)
{
    nquad qj ; 
    qj.f = m_srcnodes->getVQuad(idx, j) ;
    return qj ;
}



nmat4pair* NCSGData::import_transform_pair(unsigned itra)
{
    // itra is a 1-based index, with 0 meaning None

    assert(NTRAN == 2);
    if(itra == 0 || m_transforms == NULL ) return NULL ; 

    unsigned idx = itra - 1 ; 

    assert( idx < m_num_transforms );
    assert(m_transforms->hasShape(-1,NTRAN,4,4));

    nmat4pair* pair = m_transforms->getMat4PairPtr(idx);

    return pair ; 
}

nmat4triple* NCSGData::import_transform_triple(unsigned itra)
{
    // itra is a 1-based index, with 0 meaning None

    assert(NTRAN == 3);
    if(itra == 0 || m_transforms == NULL ) return NULL ; 

    unsigned idx = itra - 1 ; 

    assert( idx < m_num_transforms );
    assert(m_transforms->hasShape(-1,NTRAN,4,4));

    nmat4triple* triple = m_transforms->getMat4TriplePtr(idx);

    return triple ; 
}

// this is invoked from NCSG::import (constructing node tree from the buffers) 
unsigned NCSGData::addUniqueTransform( const nmat4triple* gtransform )
{
    NPY<float>* gtmp = NPY<float>::make(1,NTRAN,4,4);
    gtmp->zero();
    gtmp->setMat4Triple( gtransform, 0);

    unsigned gtransform_idx = 1 + m_gtransforms->addItemUnique( gtmp, 0 ) ; 
    delete gtmp ; 

    return gtransform_idx ; 
}


void NCSGData::dump_gtransforms() const 
{
    unsigned ni =  m_gtransforms ? m_gtransforms->getNumItems() : 0  ;

    for(unsigned i=0 ; i < ni ; i++)
    {
        const nmat4triple* u_gtran = m_gtransforms->getMat4TriplePtr(i);
        std::cout 
                  << "[" << std::setw(2) << i << "] " 
                  << *u_gtran 
                  << std::endl ; 
    }
}


void NCSGData::getSrcPlanes(std::vector<glm::vec4>& _planes, unsigned idx, unsigned num_plane ) const 
{
    assert( idx < m_num_planes );
    assert( idx + num_plane - 1 < m_num_planes );

    assert( m_srcplanes->hasShape(-1,4) );

    for(unsigned i=idx ; i < idx + num_plane ; i++)
    {
        glm::vec4 plane = m_srcplanes->getQuad(i) ;

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



