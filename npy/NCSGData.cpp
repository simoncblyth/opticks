/*
 * Copyright (c) 2019 Opticks Team. All Rights Reserved.
 *
 * This file is part of Opticks
 * (see https://bitbucket.org/simoncblyth/opticks).
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); 
 * you may not use this file except in compliance with the License.  
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software 
 * distributed under the License is distributed on an "AS IS" BASIS, 
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
 * See the License for the specific language governing permissions and 
 * limitations under the License.
 */

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
#include "nmat4triple.hpp"
#include "NCSGData.hpp"
#include "GLMFormat.hpp"

#define TREE_NODES(height) ( (0x1 << (1+(height))) - 1 )


const plog::Severity NCSGData::LEVEL = PLOG::EnvLevel("NCSGData", "DEBUG"); 


const NPYSpecList* NCSGData::MakeSPECS()
{
    NPYSpecList* sl = new NPYSpecList(); 

    int verbosity = 0 ; 
    //                                               name                  ni nj nk nl nm  type            ctrl optional verbosity         
    sl->add( (unsigned)SRC_NODES       , new NPYSpec("srcnodes.npy"       , 0, 4, 4, 0, 0, NPYBase::FLOAT , "" , false, verbosity));
    sl->add( (unsigned)SRC_IDX         , new NPYSpec("srcidx.npy"         , 0, 4, 0, 0, 0, NPYBase::UINT  , "" , false, verbosity));
    sl->add( (unsigned)SRC_TRANSFORMS  , new NPYSpec("srctransforms.npy"  , 0, 4, 4, 0, 0, NPYBase::FLOAT , "" , false, verbosity));
    sl->add( (unsigned)SRC_PLANES      , new NPYSpec("srcplanes.npy"      , 0, 4, 0, 0, 0, NPYBase::FLOAT , "" , true , verbosity));


    sl->add( (unsigned)SRC_FACES       , new NPYSpec("srcfaces.npy"       , 0, 4, 0, 0, 0, NPYBase::INT   , "" , true , verbosity));
     // analytic/prism.py _get_faces using 4       


    //                                                                       ^^^^^^  srcfaces is glm::ivec4  in nconvexpolyhedron ?????
    sl->add( (unsigned)SRC_VERTS       , new NPYSpec("srcverts.npy"       , 0, 3, 0, 0, 0, NPYBase::FLOAT , "" , true , verbosity));
    //                                                                       ^^^^^^  srcverts is glm::vec3  in nconvexpolyhedron

    sl->add( (unsigned)NODES           , new NPYSpec("nodes.npy"          , 0, 4, 4, 0, 0, NPYBase::FLOAT , "" , false, verbosity));
    sl->add( (unsigned)PLANES          , new NPYSpec("planes.npy"         , 0, 4, 0, 0, 0, NPYBase::FLOAT , "" , false, verbosity));
    sl->add( (unsigned)TRANSFORMS      , new NPYSpec("transforms.npy"     , 0, 3, 4, 4, 0, NPYBase::FLOAT , "" , false, verbosity));
    sl->add( (unsigned)GTRANSFORMS     , new NPYSpec("gtransforms.npy"    , 0, 3, 4, 4, 0, NPYBase::FLOAT , "" , false, verbosity));
    sl->add( (unsigned)IDX             , new NPYSpec("idx.npy"            , 0, 4, 0, 0, 0, NPYBase::UINT  , "" , false, verbosity));

    return sl ; 
}
const NPYSpecList* NCSGData::SPECS = MakeSPECS() ; 

NPY<float>*    NCSGData::getNodeBuffer() const {         return dynamic_cast<NPY<float>*>(m_npy->getBuffer((int)NODES)) ; }
NPY<float>*    NCSGData::getTransformBuffer() const {    return dynamic_cast<NPY<float>*>(m_npy->getBuffer((int)TRANSFORMS)) ; }
NPY<float>*    NCSGData::getGTransformBuffer() const {   return dynamic_cast<NPY<float>*>(m_npy->getBuffer((int)GTRANSFORMS)) ; }
NPY<float>*    NCSGData::getPlaneBuffer() const {        return dynamic_cast<NPY<float>*>(m_npy->getBuffer((int)PLANES)) ; }
NPY<unsigned>* NCSGData::getIdxBuffer() const {          return dynamic_cast<NPY<unsigned>*>(m_npy->getBuffer((int)IDX)) ; }

NPY<float>*    NCSGData::getSrcTransformBuffer() const { return dynamic_cast<NPY<float>*>(m_npy->getBuffer((int)SRC_TRANSFORMS)) ; }
NPY<float>*    NCSGData::getSrcNodeBuffer() const {      return dynamic_cast<NPY<float>*>(m_npy->getBuffer((int)SRC_NODES)) ; } 
NPY<float>*    NCSGData::getSrcPlaneBuffer() const {     return dynamic_cast<NPY<float>*>(m_npy->getBuffer((int)SRC_PLANES)) ; }
NPY<float>*    NCSGData::getSrcVertsBuffer() const {     return dynamic_cast<NPY<float>*>(m_npy->getBuffer((int)SRC_VERTS)) ; }
NPY<int>*      NCSGData::getSrcFacesBuffer() const {     return dynamic_cast<NPY<int>*>(m_npy->getBuffer((int)SRC_FACES)) ; }
NPY<unsigned>* NCSGData::getSrcIdxBuffer() const {       return dynamic_cast<NPY<unsigned>*>(m_npy->getBuffer((int)SRC_IDX)) ; } 


void NCSGData::init_buffers(unsigned height)  // invoked from NCSG::NCSG(nnode* root ) ie when adopting 
{

    m_height = height ; 
    unsigned num_nodes = NumNodes(height); // number of nodes for a complete binary tree of the needed height, with no balancing 
    m_num_nodes = num_nodes ; 
    LOG(LEVEL) 
        << " m_height " << m_height
        << " m_num_nodes " << m_num_nodes
        ; 

    bool zero = true ; 
    const char* msg = "init_buffer.adopt" ;  

    m_npy->initBuffer( (int)SRC_NODES     ,  m_num_nodes, zero , msg ); 
    m_npy->initBuffer( (int)SRC_TRANSFORMS,            0, zero , msg ); 
    m_npy->initBuffer( (int)SRC_PLANES    ,            0, zero , msg ); 
    m_npy->initBuffer( (int)SRC_IDX       ,            1, zero , msg ); 

    m_npy->initBuffer( (int)SRC_VERTS     ,            0, zero , msg ); 
    m_npy->initBuffer( (int)SRC_FACES     ,            0, zero , msg ); 
}

void NCSGData::loadsrc(const char* treedir)  // invoked from NCSG::NCSG(const char* treedir) , ie when loading 
{
    m_npy->loadBuffer( treedir,(int)SRC_NODES ); 
    m_npy->loadBuffer( treedir,(int)SRC_TRANSFORMS ) ; 
    m_npy->loadBuffer( treedir,(int)SRC_PLANES ); 
    m_npy->loadBuffer( treedir,(int)SRC_IDX); 

    m_npy->loadBuffer( treedir,(int)SRC_VERTS ); 
    m_npy->loadBuffer( treedir,(int)SRC_FACES ); 


    m_num_nodes = m_npy->getNumItems((int)SRC_NODES); 
    m_height = CompleteTreeHeight( m_num_nodes ) ; 

    import_src_identity();

    if(m_verbosity > 1)
        LOG(info) 
              << " verbosity(>1) " << m_verbosity  
              << smry() 
              ;
}


void NCSGData::import_src_identity()
{
    NPY<unsigned>* idx = dynamic_cast<NPY<unsigned>*>(m_npy->getBuffer((int)SRC_IDX)) ; 
    assert( idx->hasShape(1,4) ); 
    glm::uvec4 uidx = idx->getQuad_(0) ;  

    m_src_index = uidx.x ; 
    m_src_soIdx = uidx.y ; 
    m_src_lvIdx = uidx.z ; 
    m_src_height = uidx.w ; 


    bool match_height = m_src_height == m_height ;
  
    if(!match_height)
        LOG(fatal)  
            << " src_index " << m_src_index
            << " src_soIdx " << m_src_soIdx
            << " src_lvIdx " << m_src_lvIdx
            << " src_height " << m_src_height
            << " m_height " << m_height
            << ( match_height ? " MATCH " : " MISMATCH " ) 
            << "height" 
            ; 
     
    assert( match_height );


}


void NCSGData::savesrc(const char* treedir ) const 
{
    m_npy->saveBuffer( treedir,(int)SRC_NODES ); 
    m_npy->saveBuffer( treedir,(int)SRC_TRANSFORMS ); 
    m_npy->saveBuffer( treedir,(int)SRC_PLANES ); 
    m_npy->saveBuffer( treedir,(int)SRC_IDX ); 

    m_npy->saveBuffer( treedir,(int)SRC_VERTS ); 
    m_npy->saveBuffer( treedir,(int)SRC_FACES ); 
}

/**
NCSGData::prepareForImport
---------------------------

Importing buffers into the node tree requires:

1. tripletized the m_srctransforms into m_transforms

**/

void NCSGData::prepareForImport()
{
    assert(NTRAN == 3);
    NPY<float>* src = getSrcTransformBuffer(); 
    assert( src && "srctransforms buffer is required by prepareForImport "); 
    unsigned ni = src->getNumItems();  

    NPY<float>* transforms = NPY<float>::make_triple_transforms(src) ;
    assert(transforms->hasShape(ni,NTRAN,4,4));

    m_npy->setBuffer( (int)TRANSFORMS, transforms , "prepareForImport"); 
}

/**
NCSGData::prepareForExport of node tree into buffers
-------------------------------------------------------

Requires : nodes, planes buffers ready to receive the export 

**/

void NCSGData::prepareForExport()
{
    //                  bid                 ni          zero    msg        
    m_npy->initBuffer( (int)NODES         , m_num_nodes, true , "prepareForExport");
    m_npy->initBuffer( (int)PLANES        ,           0, true , "prepareForExport");
    m_npy->initBuffer( (int)IDX           ,           1, true , "prepareForExport");
}

/**
NCSGData::prepareForSetup
---------------------------

1. gtransforms buffers ready to collect unique global transforms

**/

void NCSGData::prepareForGTransforms(bool locked)
{
    bool zero = true ; 
    m_npy->initBuffer( (int)GTRANSFORMS   ,           0, zero , "prepareForSetup"); 
    m_npy->setLocked( (int)GTRANSFORMS, locked ); 
}

/**
NCSGData::addUniqueTransform
------------------------------

Used global transforms are collected into the GTransforms
buffer and the 1-based index to the transforms is returned. 
This is invoked from NCSG::addUniqueTransform

**/

unsigned NCSGData::addUniqueTransform( const nmat4triple* gtransform )
{
    NPY<float>* gtmp = NPY<float>::make(1,NTRAN,4,4);
    gtmp->zero();
    gtmp->setMat4Triple( gtransform, 0);

    NPY<float>* gtransforms = getGTransformBuffer();
    assert(gtransforms);
    unsigned gtransform_idx = 1 + gtransforms->addItemUnique( gtmp, 0 ) ; 
    delete gtmp ; 

    return gtransform_idx ; 
}





/**
NCSGData::import_transform_triple
-----------------------------------

To save manipulations on GPU matrix inverses and transposes
are done for all the transforms ahead of time.

**/

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




//////////////////////////////////////////////////////////////////////////////////////////////////////


NCSGData::NCSGData()  
    :
    m_verbosity(1),
    m_npy(new NPYList(NCSGData::SPECS)),
    m_height(0),
    m_num_nodes(0),
    m_src_index(0), 
    m_src_soIdx(0),
    m_src_lvIdx(0),
    m_src_height(0)
{
}

unsigned NCSGData::getSrcIndex() const { return m_src_index ; }
unsigned NCSGData::getSrcLVIdx() const { return m_src_lvIdx ; }
unsigned NCSGData::getSrcSOIdx() const { return m_src_soIdx ; }
unsigned NCSGData::getSrcHeight() const { return m_src_height ; }
 

NPYList* NCSGData::getNPYList() const 
{
    return m_npy ; 
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

unsigned NCSGData::NumNodes(unsigned height) // static
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



/**
NCSGData::setIdx
------------------

Used by NCSG::export_idx NCSG::export_srcidx

The (1,4) idx_buffer integers written by this are then used by 
GParts::Make when creating a GParts instance from the NCSG, 
and then this identity information gets concatenated in GParts::Combine

**/

void NCSGData::setIdx( unsigned index, unsigned soIdx, unsigned lvIdx, unsigned height, bool src )
{
    assert( height == m_height ); 
    glm::uvec4 uidx(index, soIdx, lvIdx, height); 

    NPY<unsigned>* _idx = src ? getSrcIdxBuffer() : getIdxBuffer() ;  
    assert(_idx); 
    _idx->setQuad(uidx, 0u );     

    LOG(debug) 
        << " index " << index 
        << " soIdx " << soIdx
        << " lvIdx " << lvIdx
        << " height " << height
        << ( src ? " srcIdx " : " Idx " )
        ;

}

std::string NCSGData::smry() const 
{
    std::stringstream ss ; 
    ss 
       << " ht " << std::setw(2) << m_height 
       << " nn " << std::setw(4) << m_num_nodes
       << " sid " << std::setw(4) << m_src_index
       << " sso " << std::setw(4) << m_src_soIdx
       << " slv " << std::setw(4) << m_src_lvIdx
       << " sht " << std::setw(4) << m_src_height
       << " " << m_npy->desc()
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
        glm::vec4 plane = srcplanes->getQuad_(i) ;

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




