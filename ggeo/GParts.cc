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

#include <map>
#include <iomanip>
#include <cstdio>
#include <cassert>
#include <sstream>
#include <climits>
#include <csignal>


#include "BStr.hh"
#include "BFile.hh"
#include "SPack.hh"
#include "OpticksCSG.h"

// npy-

#include "NCSG.hpp"
#include "NBBox.hpp"
#include "NGLMExt.hpp"
#include "nmat4triple.hpp"
#include "NPY.hpp"
#include "NSlice.hpp"
#include "NPart.hpp"
#include "NCSG.hpp"
#include "NQuad.hpp"
#include "NNode.hpp"
#include "NPlane.hpp"
#include "GLMFormat.hpp"

#include "GVector.hh"
#include "GItemList.hh"
#include "GBndLib.hh"
#include "GMatrix.hh"

#include "GPts.hh"
#include "GPt.hh"
#include "GParts.hh"

#include "Opticks.hh"

#include "PLOG.hh"


const plog::Severity GParts::LEVEL = PLOG::EnvLevel("GParts", "DEBUG") ; 

const char* GParts::CONTAINING_MATERIAL = "CONTAINING_MATERIAL" ;  
const char* GParts::SENSOR_SURFACE = "SENSOR_SURFACE" ;  


/**
GParts::Compare
----------------

See GPtsTest

**/


int GParts::Compare(const GParts* a, const GParts* b, bool dump )
{
    int rc = 0 ; 
    unsigned w = 34 ; 
    
    if(dump)
    std::cout 
        << std::setw(w) << "qty" 
        << std::setw(w) << "A" 
        << std::setw(w) << "B"
        << std::endl 
        ;  


    {
        unsigned av = a->getVolumeIndex(0) ; 
        unsigned bv = b->getVolumeIndex(0) ; 
        

        rc += ( av == bv ? 0 : 1 );  

        if(dump)
        std::cout 
            << std::setw(w) << "VolumeIndex"
            << std::setw(w) << av
            << std::setw(w) << bv
            << std::endl 
            ;  
    } 


    const char* aname = a->getName(); 
    const char* bname = b->getName(); 
    if(dump)
    std::cout 
        << std::setw(w) << "Name" 
        << std::setw(w) << ( aname ? aname : "NULL" )
        << std::setw(w) << ( bname ? bname : "NULL" )
        << std::endl 
        ;  

    if(dump)
    std::cout 
        << std::setw(w) << "BndLib" 
        << std::setw(w) << a->getBndLib() 
        << std::setw(w) << b->getBndLib() 
        << std::endl 
        ;  

    if(dump)
    std::cout 
        << std::setw(w) << "Closed" 
        << std::setw(w) << a->isClosed() 
        << std::setw(w) << b->isClosed() 
        << std::endl 
        ;  

    if(dump)
    std::cout 
        << std::setw(w) << "Loaded" 
        << std::setw(w) << a->isLoaded() 
        << std::setw(w) << b->isLoaded() 
        << std::endl 
        ;  
 
    if(dump)
    std::cout 
        << std::setw(w) << "PrimFlagString" 
        << std::setw(w) << a->getPrimFlagString() 
        << std::setw(w) << b->getPrimFlagString() 
        << std::endl 
        ;  
   
    if(dump)
    std::cout 
        << std::setw(w) << "NumParts" 
        << std::setw(w) << a->getNumParts() 
        << std::setw(w) << b->getNumParts() 
        << std::endl 
        ;  

    if(dump)
    std::cout 
        << std::setw(w) << "NumPrim" 
        << std::setw(w) << a->getNumPrim() 
        << std::setw(w) << b->getNumPrim() 
        << std::endl 
        ;  


    std::vector<const char*> tags = {"prim", "idx", "part", "tran", "plan" }; 
    for(unsigned i=0 ; i < tags.size() ; i++)
    {
        const char* tag = tags[i]; 

        std::string av = a->getBufferBase(tag)->getDigestString() ;
        std::string bv = b->getBufferBase(tag)->getDigestString() ;
    
        rc += ( av.compare(bv) == 0 ? 0 : 1 ) ; 

        if(dump)
        std::cout 
            << std::setw(w) << tags[i] 
            << std::setw(w) << av 
            << std::setw(w) << bv
            << std::endl 
            ;  
    } 
    return rc ; 
}


/**
GParts::Create from GPts
--------------------------

Canoically invoked from ``GGeo::deferredCreateGParts``
by ``GGeo::postLoadFromCache`` or ``GGeo::postDirectTranslation``.

The (GPt)pt from each GVolume yields a per-volume (GParts)parts instance
that is added to the (GParts)com instance.

``GParts::Create`` from ``GPts`` duplicates the standard precache GParts 
in a deferred postcache manner using NCSG solids persisted with GMeshLib 
and the requisite GParts arguments (spec, placement transforms) persisted by GPts 
together with the GGeoLib merged meshes.  

Note that GParts::applyPlacementTransform is applied to each individual 
GParts object prior to combination into a composite GParts using the placement 
transform collected into the GPt objects transported via GPts.

GMergedMesh::mergeVolume
GMergedMesh::mergeVolumeAnalytic
     combining and applying placement transform

* GPts instances for each mergedMesh and merged from individual volume GPts. 

* testing this with GPtsTest, using GParts::Compare 

**/


int GParts::DEBUG = -1 ; 
void GParts::SetDEBUG(int dbg)
{
    DEBUG = dbg ; 
}



GParts* GParts::Create(const Opticks* ok, const GPts* pts, const std::vector<const NCSG*>& solids, unsigned* num_mismatch_pt, std::vector<glm::mat4>* mismatch_placements ) // static
{
    plog::Severity level = DEBUG == 0 ? LEVEL : info ;  

    unsigned num_pt = pts->getNumPt(); 

    LOG(level) 
         << "[  deferred creation from GPts" 
         << " DEBUG " << DEBUG
         << " level " << level << " " << PLOG::_name(level)
         << " LEVEL " << LEVEL << " " << PLOG::_name(LEVEL)
         << " num_pt " << num_pt 
         ; 

    GParts* com = new GParts() ; 
    com->setOpticks(ok); 


    unsigned verbosity = 0 ; 
    std::vector<unsigned> mismatch_pt ; 

    for(unsigned i=0 ; i < num_pt ; i++)
    {
        const GPt* pt = pts->getPt(i); 
        int   lvIdx = pt->lvIdx ; 
        int   ndIdx = pt->ndIdx ; 
        const std::string& spec = pt->getSpec() ; 
        const glm::mat4& placement = pt->getPlacement() ; 

        LOG(level) 
            << " pt " << std::setw(4) 
            << " lv " << std::setw(4) << lvIdx 
            << " nd " << std::setw(6) << ndIdx 
            << " pl " << GLMFormat::Format(placement)
            << " bn " << spec
            ;

        assert( lvIdx > -1 ); 

        const NCSG* csg = unsigned(lvIdx) < solids.size() ? solids[lvIdx] : NULL ; 
        assert( csg ); 

        //  X4PhysicalVolume::convertNode

        GParts* parts = GParts::Make( csg, spec.c_str(), ndIdx ); 

        unsigned num_mismatch = 0 ; 

        parts->applyPlacementTransform( placement, verbosity, num_mismatch );   // this changes parts:m_tran_buffer

        if(num_mismatch > 0 )
        {
            LOG(error) << " pt " << i << " invert_trs num_mismatch : " << num_mismatch ; 
            mismatch_pt.push_back(i); 
            if(mismatch_placements)
            {
                glm::mat4 placement_(placement);
                placement_[0][3] = SPack::unsigned_as_float(i);  
                placement_[1][3] = SPack::unsigned_as_float(lvIdx);  
                placement_[2][3] = SPack::unsigned_as_float(ndIdx);  
                placement_[3][3] = SPack::unsigned_as_float(num_mismatch);  
                mismatch_placements->push_back(placement_); 
            }
        }

        parts->dumpTran("parts"); 
        com->add( parts ); 
    }

    com->dumpTran("com");

    if(num_mismatch_pt) 
    {
        *num_mismatch_pt = mismatch_pt.size(); 
        if(*num_mismatch_pt > 0)
        {
            LOG(error) << "num_mismatch_pt : " << *num_mismatch_pt ; 
            std::cout << " mismatch_pt indices : " ; 
            for(unsigned i=0 ; i < *num_mismatch_pt ; i++) std::cout << mismatch_pt[i] << " " ; 
            std::cout << std::endl ; 
        }
    }

    LOG(level) << "]" ; 
    return com ; 
}


const std::vector<GParts*>& GParts::getSubs() const 
{
    return m_subs ; 
}
unsigned GParts::getNumSubs() const 
{
    return m_subs.size(); 
}
GParts* GParts::getSub(unsigned i) const 
{
    assert( i < m_subs.size() ); 
    return m_subs[i] ; 
}

void GParts::setRepeatIndex(unsigned ridx) 
{
    m_ridx = ridx ; 
}  
unsigned GParts::getRepeatIndex() const 
{
    return m_ridx ; 
}






/**
GParts::Combine
------------------

Concatenate vector of GParts instances into a single GParts instance

* getPrimFlag of all instances must be the same, typically CSG_FLAGNODETREE
  and not the old CSG_FLAGPARTLIST that predates full CSG, and is adopted 
  by the combined GParts

* getAnalyticVersion of all instances is also required to be the same and is
  adopted by the combined GParts

* the first GBndLib encountered is adopted for the combined GParts

* concatenation is done with GParts::add

Currently GParts::Combine is used only from tests:

* extg4/tests/X4PhysicalVolume2Test.cc
* extg4/tests/X4SolidTest.cc

for example GMergedMesh::mergeMergedMesh uses GParts::add (as does this) 
rather than using this higher level function.

**/

GParts* GParts::Combine(std::vector<GParts*> subs)  // static
{
    LOG(LEVEL) << "[ " << subs.size() ; 

    GParts* parts = new GParts(); 

    GBndLib* bndlib = NULL ; 
    unsigned analytic_version = 0 ;  
    OpticksCSG_t primflag = CSG_ZERO ; 


    for(unsigned int i=0 ; i < subs.size() ; i++)
    {
        GParts* sp = subs[i];

        OpticksCSG_t pf = sp->getPrimFlag();


        if(primflag == CSG_ZERO) 
            primflag = pf ; 
        else
            assert(pf == primflag && "GParts::combine requires all GParts instances to have the same primFlag (either CSG_FLAGNODETREE or legacy CSG_FLAGPARTLIST)" );


        unsigned av = sp->getAnalyticVersion();

        if(analytic_version == 0)
            analytic_version = av ;
        else
            assert(av == analytic_version && "GParts::combine requires all GParts instances to have the same analytic_version " );   


        parts->add(sp);

        if(!bndlib) bndlib = sp->getBndLib(); 
    } 

    if(bndlib) parts->setBndLib(bndlib);
    parts->setAnalyticVersion(analytic_version);
    parts->setPrimFlag(primflag);

    LOG(LEVEL) << "]" ; 
    return parts ; 
}





GParts* GParts::Combine(GParts* onesub)  // static
{
    // for consistency: need to combine even when only one sub
    std::vector<GParts*> subs ; 
    subs.push_back(onesub); 
    return GParts::Combine(subs);
}


/**
GParts::Make
-------------

Serialize the npart shape (BOX, SPHERE or PRISM) into a (1,4,4) parts buffer. 
Then instanciate a GParts instance to hold the parts buffer 
together with the boundary spec.

**/

GParts* GParts::Make(const npart& pt, const char* spec)
{
    NPY<unsigned>* idxBuf = NPY<unsigned>::make(1,4);
    idxBuf->zero();

    NPY<float>* partBuf = NPY<float>::make(1, NJ, NK );
    partBuf->zero();

    NPY<float>* tranBuf = NPY<float>::make(0, NTRAN, 4, 4 );
    tranBuf->zero();

    NPY<float>* planBuf = NPY<float>::make(0, 4 );
    planBuf->zero();

    partBuf->setPart( pt, 0u );

    GParts* gpt = new GParts(idxBuf, partBuf,tranBuf,planBuf,spec) ;

    unsigned typecode = gpt->getTypeCode(0u);
    assert(typecode == CSG_BOX || typecode == CSG_SPHERE || typecode == CSG_PRISM);

    return gpt ; 
}

GParts* GParts::Make(OpticksCSG_t csgflag, glm::vec4& param, const char* spec)
{
    float size = param.w ;  
    //-------   
    // FIX: this is wrong for most solids 
    //gbbox bb(gfloat3(-size), gfloat3(size));  
     
    nbbox bb = make_bbox(glm::vec3(-size), glm::vec3(size));  


    if(csgflag == CSG_ZSPHERE)
    {
        assert( 0 && "TODO: geometry specifics should live in nzsphere etc.. not here " );
        bb.min.z = param.x*param.w ; 
        bb.max.z = param.y*param.w ; 
    } 
    //-------


    NPY<unsigned>* idxBuf = NPY<unsigned>::make(1,4);
    idxBuf->zero();

    NPY<float>* partBuf = NPY<float>::make(1, NJ, NK );
    partBuf->zero();

    NPY<float>* tranBuf = NPY<float>::make(0, NTRAN, 4, 4 );
    tranBuf->zero();

    NPY<float>* planBuf = NPY<float>::make(0, 4);
    planBuf->zero();



    assert(BBMIN_K == 0 );
    assert(BBMAX_K == 0 );

    unsigned int i = 0u ; 
    partBuf->setQuad( i, PARAM_J, param.x, param.y, param.z, param.w );
    partBuf->setQuad( i, BBMIN_J, bb.min.x, bb.min.y, bb.min.z , 0.f );
    partBuf->setQuad( i, BBMAX_J, bb.max.x, bb.max.y, bb.max.z , 0.f );

    // TODO: go via an npart instance

    GParts* pt = new GParts(idxBuf, partBuf, tranBuf, planBuf, spec) ;
    pt->setTypeCode(0u, csgflag);

    return pt ; 
} 

const int GParts::NTRAN = 3 ; 

void GParts::setCSG(const NCSG* csg)
{
    m_csg = csg ; 
}
const NCSG* GParts::getCSG() const 
{
    return m_csg ; 
}




/**
GParts::Make from NCSG tree
----------------------------

This is canonically invoked from X4PhysicalVolume::convertNode 
within the recursive visit of X4PhysicalVolume::convertStructure_r
which is doing the direct translation of Geant4 geometry.

The spec string (aka boundary name) combines four names of materials
and surfaces omat/osur/isur/imat.   

The boundary name is a "node level thing" because surfaces and materials depends 
on where you are in the structural node tree.
Contrast this with "solid level quantities" which are related just to the distinct
shapes of the solids.

In general it is better for information to be mesh or solid level 
where possible because there are far fewer distinct meshes/solids in a geometry 
that there are nodes in the structural tree.

Because GParts requires the boundary spec are forced to create it 
at node level.


Promotion from mesh/solid level to node/volume level : remember to clone
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When objects are "usedglobally" and need an independent non-instanced
identity (ie they need to be changed) then it is imperative to clone them.
Forgetting to clone leads to bizarre bugs such as duplicated indices, 
see notes/issues/GPtsTest.rst  

If any convexpolyhedron eg Trapezoids are usedglobally (ie non-instanced), will need:

1. clone the PlaneBuffer 
2. transform the planes with the global transform, do this in applyPlacementTransform 

**/

GParts* GParts::Make( const NCSG* tree, const char* spec, unsigned ndIdx )
{
    assert(spec);

    bool usedglobally = tree->isUsedGlobally() ;   // see opticks/notes/issues/subtree_instances_missing_transform.rst
    assert( usedglobally == true );  // always true now ?   

    NPY<unsigned>* tree_idxbuf = tree->getIdxBuffer() ;   // (1,4) identity indices (index,soIdx,lvIdx,height)
    NPY<float>*   tree_tranbuf = tree->getGTransformBuffer() ;
    NPY<float>*   tree_planbuf = tree->getPlaneBuffer() ;
    assert( tree_tranbuf );

    NPY<unsigned>* idxbuf = tree_idxbuf->clone()  ;   // <-- lacking this clone was cause of the mystifying repeated indices see notes/issues/GPtsTest             
    NPY<float>* nodebuf = tree->getNodeBuffer();       // serialized binary tree
    NPY<float>* tranbuf = usedglobally                 ? tree_tranbuf->clone() : tree_tranbuf ; 
    NPY<float>* planbuf = usedglobally && tree_planbuf ? tree_planbuf->clone() : tree_planbuf ;  

     
    // overwrite the cloned idxbuf swapping the tree index for the ndIdx 
    // as being promoted to node level 
    {
        assert( idxbuf->getNumItems() == 1 ); 
        unsigned i=0u ; 
        unsigned j=0u ; 
        unsigned k=0u ; 
        unsigned l=0u ; 
        idxbuf->setUInt(i,j,k,l, ndIdx);
    }

    unsigned verbosity = tree->getVerbosity(); 
    if(verbosity > 1)
    LOG(info) 
        << " tree " << std::setw(5) << tree->getIndex()
        << " usedglobally " << std::setw(1) << usedglobally 
        << " nodebuf " << ( nodebuf ? nodebuf->getShapeString() : "NULL" ) 
        << " tranbuf " << ( tranbuf ? tranbuf->getShapeString() : "NULL" ) 
        << " planbuf " << ( planbuf ? planbuf->getShapeString() : "NULL" ) 
        ;

    if(!tranbuf) 
    {
       LOG(fatal) << "GParts::make NO GTransformBuffer " ; 
       assert(0);

       tranbuf = NPY<float>::make(0,NTRAN,4,4) ;
       tranbuf->zero();
    } 
    assert( tranbuf && tranbuf->hasShape(-1,NTRAN,4,4));

    if(!planbuf) 
    {
       planbuf = NPY<float>::make(0,4) ;
       planbuf->zero();
    } 
    assert( planbuf && planbuf->hasShape(-1,4));

    nnode* root = tree->getRoot(); 
    // hmm maybe should not use the nnode ? ie operate fully from the persistable buffers ?

    assert(nodebuf && root) ; 

    unsigned ni = nodebuf->getShape(0);
    assert( nodebuf->hasItemShape(NJ, NK) && ni > 0 );

    bool type_ok = root && root->type < CSG_UNDEFINED ;
    if(!type_ok)
        LOG(fatal) << "GParts::make"
                   << " bad type " << root->type
                   << " name " << CSG::Name(root->type) 
                   << " YOU MAY JUST NEED TO RECOMPILE " 
                   ;

    assert(type_ok);

    LOG(debug) << "GParts::make NCSG "
              << " treedir " << tree->getTreeDir()
              << " node_sh " << nodebuf->getShapeString()
              << " tran_sh " << tranbuf->getShapeString() 
              << " spec " << spec 
              << " type " << root->csgname()
              ; 

    // GParts originally intended to handle lists of parts each of which 
    // must have an associated boundary spec. When holding CSG trees there 
    // is really only a need for a single common boundary, but for
    // now enable reuse of the old GParts by duplicating the spec 
    // for every node of the tree

    const char* reldir = "" ;  // empty reldir avoids defaulting to GItemList  

    GItemList* lspec = GItemList::Repeat("GParts", spec, ni, reldir) ; 

    GParts* pts = new GParts(idxbuf, nodebuf, tranbuf, planbuf, lspec) ;

    //pts->setTypeCode(0u, root->type);   //no need, slot 0 is the root node where the type came from

    pts->setCSG(tree); 

    return pts ; 
}


#ifdef GPARTS_DEBUG
std::vector<unsigned>* GParts::IDXS = NULL ;

void GParts::initDebugDupeIdx()
{
    if(IDXS == NULL) IDXS = new std::vector<unsigned> ; 
    if( m_idx_buffer && m_idx_buffer->getNumItems() == 1 )
    {
        glm::uvec4 uv = m_idx_buffer->getQuadU(0) ; 
        bool dupe = std::find(IDXS->begin(), IDXS->end(), uv.x ) != IDXS->end()  ;   
        if(dupe)
        {
           LOG(fatal) 
                << " ctor dupe " << glm::to_string( uv ) 
                << " count " << IDXS->size()
                ;   
        } 
        assert(!dupe); 
        IDXS->push_back(uv.x); 
    } 
}
#endif


GParts::GParts(GBndLib* bndlib) 
    :
    m_idx_buffer(NPY<unsigned>::make(0, 4)),
    m_part_buffer(NPY<float>::make(0, NJ, NK )),
    m_tran_buffer(NPY<float>::make(0, NTRAN, 4, 4 )),
    m_plan_buffer(NPY<float>::make(0, 4)),
    m_bndspec(new GItemList("GParts","")),   // empty reldir allows GParts.txt to be written directly at eg GPartsAnalytic/0/GParts.txt
    m_bndlib(bndlib),
    m_name(NULL),
    m_prim_buffer(NULL),
    m_closed(false),
    m_loaded(false),
    m_verbosity(0),
    m_analytic_version(0),
    m_primflag(CSG_FLAGNODETREE),
    m_medium(NULL),
    m_csg(NULL),
    m_ridx(~0u),
    m_ok(bndlib ? bndlib->getOpticks(): nullptr)
{
    m_idx_buffer->zero();
    m_part_buffer->zero();
    m_tran_buffer->zero();
    m_plan_buffer->zero();
 
    init() ; 
}


GParts::GParts(NPY<unsigned>* idxBuf, NPY<float>* partBuf,  NPY<float>* tranBuf, NPY<float>* planBuf, const char* spec, GBndLib* bndlib) 
    :
    m_idx_buffer(idxBuf ? idxBuf : NPY<unsigned>::make(0, 4)),
    m_part_buffer(partBuf ? partBuf : NPY<float>::make(0, NJ, NK )),
    m_tran_buffer(tranBuf ? tranBuf : NPY<float>::make(0, NTRAN, 4, 4 )),
    m_plan_buffer(planBuf ? planBuf : NPY<float>::make(0, 4)),
    m_bndspec(new GItemList("GParts","")),   // empty reldir allows GParts.txt to be written directly at eg GPartsAnalytic/0/GParts.txt
    m_bndlib(bndlib),
    m_name(NULL),
    m_prim_buffer(NULL),
    m_closed(false),
    m_loaded(false),
    m_verbosity(0),
    m_analytic_version(0),
    m_primflag(CSG_FLAGNODETREE),
    m_medium(NULL),
    m_csg(NULL),
    m_ridx(~0u),
    m_ok(bndlib ? bndlib->getOpticks() : nullptr)
{
    m_bndspec->add(spec);

    init() ; 
}
GParts::GParts(NPY<unsigned>* idxBuf, NPY<float>* partBuf,  NPY<float>* tranBuf, NPY<float>* planBuf, GItemList* spec, GBndLib* bndlib) 
    :
    m_idx_buffer(idxBuf ? idxBuf : NPY<unsigned>::make(0, 4)),
    m_part_buffer(partBuf ? partBuf : NPY<float>::make(0, NJ, NK )),
    m_tran_buffer(tranBuf ? tranBuf : NPY<float>::make(0, NTRAN, 4, 4 )),
    m_plan_buffer(planBuf ? planBuf : NPY<float>::make(0, 4)),
    m_bndspec(spec),
    m_bndlib(bndlib),
    m_name(NULL),
    m_prim_buffer(NULL),
    m_closed(false),
    m_loaded(false),
    m_verbosity(0),
    m_analytic_version(0),
    m_primflag(CSG_FLAGNODETREE),
    m_medium(NULL),
    m_csg(NULL),
    m_ridx(~0u),
    m_ok(bndlib ? bndlib->getOpticks() : nullptr)
{
    checkSpec(spec); 
    init() ; 
}


/**
GParts::checkSpec
-----------------

RelDir is GItemList ctor argument which 
GPmt::loadFromCache plants the relative PmtPath in ? 

The assert is tripped by GPmtTest without the special casing 

**/

void GParts::checkSpec(GItemList* spec) const 
{
    const std::string& reldir = spec->getRelDir() ;
    bool empty_rel = reldir.empty() ;
    bool is_gpmt = !empty_rel && reldir.find("GPmt/") == 0 ; 
    if(is_gpmt)
    {
        LOG(info) << "is_gpmt " << reldir ; 
    } 
    else
    {
        if(!empty_rel)
            LOG(fatal)
                << " EXPECTING EMPTY RelDir FOR NON GPmt GParts [" << reldir << "]"
                ;
        assert( empty_rel );
    }
}



void GParts::init()
{
    unsigned npart = m_part_buffer ? m_part_buffer->getNumItems() : 0 ;
    unsigned nspec = m_bndspec ? m_bndspec->getNumItems() : 0  ;

    bool match = npart == nspec ; 
    if(!match) 
    LOG(fatal) 
        << " parts/spec MISMATCH "  
        << " npart " << npart 
        << " nspec " << nspec
        ;

    assert(match);
#ifdef GPARTS_DEBUG
    //initDebugDupeIdx();
#endif

}

void GParts::setPrimFlag(OpticksCSG_t primflag)
{
    assert(primflag == CSG_FLAGNODETREE || primflag == CSG_FLAGPARTLIST || primflag == CSG_FLAGINVISIBLE );
    m_primflag = primflag ; 
}

OpticksCSG_t GParts::getPrimFlag() const 
{
    return m_primflag ;
}

const char* GParts::getPrimFlagString() const 
{
    return CSG::Name(m_primflag); 
}

bool GParts::isPartList() const  // LEGACY ANALYTIC, NOT LONG TO LIVE ? ACTUALLY ITS FASTER SO BETTER TO KEEP ALIVE
{
    return m_primflag == CSG_FLAGPARTLIST ;
}
bool GParts::isNodeTree() const // ALMOST ALWAYS THIS ONE NOWADAYS
{
    return m_primflag == CSG_FLAGNODETREE ;
}
bool GParts::isInvisible() const
{
    return m_primflag == CSG_FLAGINVISIBLE ;
}


void GParts::setInvisible()
{
    setPrimFlag(CSG_FLAGINVISIBLE);
}
void GParts::setPartList()
{
    setPrimFlag(CSG_FLAGPARTLIST);
}
void GParts::setNodeTree()
{
    setPrimFlag(CSG_FLAGNODETREE);
}
void GParts::setOpticks(const Opticks* ok)
{
    m_ok = ok ; 
}








void GParts::BufferTags(std::vector<std::string>& tags) // static
{
    tags.push_back("part");
    tags.push_back("tran");
    tags.push_back("plan");
   // tags.push_back("prim");   
}

const char* GParts::BufferName(const char* tag) // static
{
    return BStr::concat(tag, "Buffer.npy", NULL) ;
}


/**
GParts::save
--------------

Canonically invoked from GGeoLib, saving merged GParts together 
with corresponding GMergedMesh. 

Also called by X4SolidTest:test_cathode, X4PhysicalVolume2Test

**/

void GParts::save(const char* dir, const char* rela)
{
    std::string path = BFile::FormPath(dir, rela); 
    save(path.c_str()); 
}

void GParts::save(const char* dir)
{
    if(!dir) return ; 

    LOG(LEVEL) << "dir " << dir ; 

    if(!isClosed())
    {
        LOG(LEVEL) << "pre-save closing, for primBuf   " ; 
        close();
    }    

    std::vector<std::string> tags ; 
    BufferTags(tags);

    for(unsigned i=0 ; i < tags.size() ; i++)
    {
        const char* tag = tags[i].c_str();
        const char* name = BufferName(tag);
        NPY<float>* buf = getBuffer(tag);
        if(buf)
        {
            unsigned num_items = buf->getShape(0);
            if(num_items > 0)
            { 
                buf->save(dir, name);     
            }
        }
    } 
    if(m_idx_buffer) m_idx_buffer->save(dir, BufferName("idx"));    
    if(m_prim_buffer) m_prim_buffer->save(dir, BufferName("prim"));    

    // TODO: see if can use NPYBase to treat all buffers uniformly using getBufferBase

    if(m_bndspec) m_bndspec->save(dir); 
}



template<typename T>
NPY<T>* GParts::LoadBuffer(const char* dir, const char* tag) // static
{
    const char* name = BufferName(tag) ;
    bool quietly = true ; 
    NPY<T>* buf = NPY<T>::load(dir, name, quietly ) ;
    return buf ; 
}

GParts* GParts::Load(const char* dir) // static
{
    LOG(debug) << "GParts::Load dir " << dir ; 

    NPY<unsigned>* idxBuf = LoadBuffer<unsigned>(dir, "idx");
    NPY<float>* partBuf = LoadBuffer<float>(dir, "part");
    NPY<float>* tranBuf = LoadBuffer<float>(dir, "tran");
    NPY<float>* planBuf = LoadBuffer<float>(dir, "plan");

    // hmm what is appropriate for spec and bndlib these ? 
    //
    // bndlib has to be externally set, its a global thing 
    // that is only needed by registerBoundaries
    //
    // spec is internal ... it needs to be saved with the GParts
    //    

    const char* reldir = "" ;   // empty, signally inplace itemlist persisting
    GItemList* bndspec = GItemList::Load(dir, "GParts", reldir ) ; 
    GBndLib*  bndlib = NULL ; 
    GParts* parts = new GParts(idxBuf, partBuf,  tranBuf, planBuf, bndspec, bndlib) ;
    
    NPY<int>* primBuf = NPY<int>::load(dir, BufferName("prim") );
    parts->setPrimBuffer(primBuf);
    parts->setLoaded();

    return parts  ; 
}



void GParts::setName(const char* name)
{
    m_name = name ? strdup(name) : NULL  ; 
}
const char* GParts::getName() const 
{
    return m_name ; 
}

void GParts::setVerbosity(unsigned verbosity)
{
    m_verbosity = verbosity ; 
}
unsigned GParts::getVerbosity() const 
{
    return m_verbosity ;
}



unsigned GParts::getAnalyticVersion()
{
    return m_analytic_version ; 
}
void GParts::setAnalyticVersion(unsigned version)
{
    m_analytic_version = version ; 
}
 



bool GParts::isClosed() const { return m_closed ; } 
bool GParts::isLoaded() const { return m_loaded ; }

void GParts::setLoaded(bool loaded)
{
    m_loaded = loaded ; 
}



unsigned int GParts::getPrimNumParts(unsigned int prim_index)
{
   // DOES NOT WORK POSTCACHE
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
GBndLib* GParts::getBndLib() const 
{
    return m_bndlib ; 
}


void GParts::setIdxBuffer(NPY<unsigned>* buf ) { m_idx_buffer = buf ; } 
void GParts::setPrimBuffer(NPY<int>* buf ) {     m_prim_buffer = buf ; } 
void GParts::setPartBuffer(NPY<float>* buf ) {   m_part_buffer = buf ; } 
void GParts::setTranBuffer(NPY<float>* buf) {    m_tran_buffer = buf ; } 
void GParts::setPlanBuffer(NPY<float>* buf) {    m_plan_buffer = buf ; } 

NPY<int>*      GParts::getPrimBuffer() const  { return m_prim_buffer ;  }
NPY<unsigned>* GParts::getIdxBuffer() const {   return m_idx_buffer ; } 
NPY<float>*    GParts::getPartBuffer() const {  return m_part_buffer ; } 
NPY<float>*    GParts::getTranBuffer() const {  return m_tran_buffer ; } 
NPY<float>*    GParts::getPlanBuffer() const {  return m_plan_buffer ; } 



unsigned GParts::getNumTran() const 
{
    return  m_tran_buffer ? m_tran_buffer->getShape(0) : 0u ; 
}

glm::mat4 GParts::getTran(unsigned tranIdx, unsigned j) const 
{
    bool inrange = tranIdx < m_tran_buffer->getShape(0)  ; 
    if(!inrange)
    {
         LOG(error) 
            << " OUT OF RANGE "
            << " m_tran_buffer " << m_tran_buffer->getShapeString()
            << " tranIdx " << tranIdx
            << " j " << j 
            ; 
    }
    assert(inrange); 
    return m_tran_buffer->getMat4_(tranIdx, j); 
}








NPYBase* GParts::getBufferBase(const char* tag) const 
{
    if(strcmp(tag,"prim")==0) return m_prim_buffer ; 
    if(strcmp(tag,"idx")==0) return m_idx_buffer ; 
    if(strcmp(tag,"part")==0) return m_part_buffer ; 
    if(strcmp(tag,"tran")==0) return m_tran_buffer ; 
    if(strcmp(tag,"plan")==0) return m_plan_buffer ; 
    return NULL ; 
}

NPY<float>* GParts::getBuffer(const char* tag) const 
{
    if(strcmp(tag,"part")==0) return m_part_buffer ; 
    if(strcmp(tag,"tran")==0) return m_tran_buffer ; 
    if(strcmp(tag,"plan")==0) return m_plan_buffer ; 
    //if(strcmp(tag,"prim")==0) return m_prim_buffer ; 
    return NULL ; 
}


unsigned GParts::getNumIdx() const 
{
    return m_idx_buffer->getNumItems() ; 
}

unsigned GParts::getNumParts() const 
{
    // for combo GParts this is total of all prim
    if(!m_part_buffer)
    {
        LOG(error) << "GParts::getNumParts NULL part_buffer" ; 
        return 0 ; 
    }

    assert(m_part_buffer->getNumItems() == m_bndspec->getNumItems() );
    return m_part_buffer->getNumItems() ;
}


/**
GParts::applyPlacementTransform
--------------------------------

1. transforms the entire m_tran_buffer with the passed transform, 
   to avoid leaving behind constituents this means that every constituent
   must have an associated transform, **even if its the identity transform**

* This was formerly invoked from GGeo::prepare...GMergedMesh::mergeVolumeAnalytic
* Now it is invoked by GParts::Create 

**/

void GParts::applyPlacementTransform(GMatrix<float>* gtransform, unsigned verbosity, unsigned& num_mismatch )
{
    const float* data = static_cast<float*>(gtransform->getPointer());
    if(verbosity > 2)
    nmat4triple::dump(data, "GParts::applyPlacementTransform gtransform:" ); 
    glm::mat4 placement = glm::make_mat4( data ) ;  

    applyPlacementTransform( placement, verbosity, num_mismatch ); 
}



/**
GParts::applyPlacementTransform
---------------------------------

Formerly all geometry that required planes (eg trapezoids) 
was part of instanced solids... so this was not needed.
BUT for debugging it is useful to be able to operate in global mode
whilst testing small subsets of geometry

**/

void GParts::applyPlacementTransform(const glm::mat4& placement, unsigned verbosity, unsigned& num_mismatch )
{
    plog::Severity level = DEBUG ? info : LEVEL ;  

    LOG(level) << "[ placement " << glm::to_string( placement ) ; 

    //std::raise(SIGINT); 

    assert(m_tran_buffer->hasShape(-1,3,4,4));

    unsigned ni = m_tran_buffer->getNumItems();

    LOG(level) 
        << " tran_buffer " << m_tran_buffer->getShapeString()
        << " ni " << ni
        ;


    bool reversed = true ; // means apply transform at root end, not leaf end 

    if(verbosity > 2 || DEBUG > 0)
    nmat4triple::dump(m_tran_buffer,"GParts::applyPlacementTransform before");


    std::vector<unsigned> mismatch ;  

    for(unsigned i=0 ; i < ni ; i++)
    {
        nmat4triple* tvq = m_tran_buffer->getMat4TriplePtr(i) ;

        bool match = true ; 
        const nmat4triple* ntvq = nmat4triple::make_transformed( tvq, placement, reversed, "GParts::applyPlacementTransform", match );
                              //  ^^^^^^^^^^^^^^^^^^^^^^^ SUSPECT DOUBLE NEGATIVE RE REVERSED  ^^^^^^^

        if(!match) mismatch.push_back(i);   

        m_tran_buffer->setMat4Triple( ntvq, i ); 
    }

   
    num_mismatch = mismatch.size(); 

    if(num_mismatch > 0 || verbosity > 2 || DEBUG > 0)
    {
        nmat4triple::dump(m_tran_buffer,"GParts::applyPlacementTransform after");

        LOG(info) << " num_mismatch " << num_mismatch ; 
        std::cout << " mismatch indices : " ; 
        for(unsigned i=0 ; i < mismatch.size() ; i++) std::cout << mismatch[i] << " " ; 
        std::cout << std::endl ; 

    }


    assert(m_plan_buffer->hasShape(-1,4));
    unsigned num_plane = m_plan_buffer->getNumItems();

    if(num_plane > 0) 
    {
        if(verbosity > 3)
        m_plan_buffer->dump("planes_before_transform");

        nglmext::transform_planes( m_plan_buffer, placement );

        if(verbosity > 3)
        m_plan_buffer->dump("planes_after_transform");
    }

    LOG(LEVEL) << "] placement " << glm::to_string( placement ) ; 
}


std::string GParts::id() const 
{
    std::stringstream ss ; 
    ss  << "GParts::id" ;

    const NCSG* csg = getCSG();
    if(csg)
    {
        unsigned soIdx = csg->getSOIdx(); 
        unsigned lvIdx = csg->getLVIdx(); 
        ss
           << " soIdx " << soIdx 
           << " lvIdx " << lvIdx 
           ;
    }    
    return ss.str(); 
}


/**
GParts::add
-------------

Basis for combination of analytic geometry.


**/

void GParts::add(GParts* other)
{
    m_subs.push_back(other); 

    if(getBndLib() == NULL)
    {
        setBndLib(other->getBndLib()); 
    }
    else
    {
        assert(getBndLib() == other->getBndLib());
    }

    unsigned int n0 = getNumParts(); // before adding

    m_bndspec->add(other->getBndSpec());


    // count the tran and plan collected so far into this GParts
    unsigned tranOffset = m_tran_buffer->getNumItems(); 
    //unsigned planOffset = m_plan_buffer->getNumItems(); 

    NPY<unsigned>* other_idx_buffer = other->getIdxBuffer() ;
    NPY<float>* other_part_buffer = other->getPartBuffer()->clone() ;
    NPY<float>* other_tran_buffer = other->getTranBuffer() ;
    NPY<float>* other_plan_buffer = other->getPlanBuffer() ;


    if(m_ok && m_ok->isGPartsTransformOffset())  // --gparts_transform_offset
    {
        LOG(LEVEL) << " --gparts_transform_offset " ; 
        bool preserve_zero = true ; 
        bool preserve_signbit = true ; 
        other_part_buffer->addOffset(GTRANSFORM_J, GTRANSFORM_K, tranOffset, preserve_zero, preserve_signbit );  
        // hmm offsetting of planes needs to be done only for parts of type CSG_CONVEXPOLYHEDRON 
    }
    else
    {
        LOG(LEVEL) << " NOT --gparts_transform_offset " ; 
    }

    m_idx_buffer->add(other_idx_buffer);
    m_part_buffer->add(other_part_buffer);
    m_tran_buffer->add(other_tran_buffer);
    m_plan_buffer->add(other_plan_buffer);

    unsigned num_idx_add = other_idx_buffer->getNumItems() ;
    assert( num_idx_add == 1);

#ifdef GPARTS_DEBUG
    glm::uvec4 idx = other_idx_buffer->getQuadU(0);  
    unsigned aix = idx.x ;  
    bool dupe = std::find(m_aix.begin(), m_aix.end(), aix ) != m_aix.end() ; 
    m_aix.push_back(aix);  

    LOG(debug) 
        << " idx " << glm::to_string(idx) 
        << " aix " << aix
        << " m_idx_buffer.NumItems " << m_idx_buffer->getNumItems()
        << " m_aix.size " << m_aix.size()
        ; 

    if(dupe) LOG(fatal) << " dupe " << aix ; 
    assert(!dupe); 
#endif

    unsigned num_part_add = other_part_buffer->getNumItems() ;
    unsigned num_tran_add = other_tran_buffer->getNumItems() ;
    unsigned num_plan_add = other_plan_buffer->getNumItems() ;

    m_part_per_add.push_back(num_part_add); 
    m_tran_per_add.push_back(num_tran_add);
    m_plan_per_add.push_back(num_plan_add);

    unsigned int n1 = getNumParts(); // after adding

    for(unsigned int p=n0 ; p < n1 ; p++)  // update indices for parts added
    {
        setIndex(p, p);
    }

    // TODO: 
    //    Transform (and plane) references in m_part_buffer 
    //    need to be offset on combination : so they can stay valid
    //    against the combined transform (and plane) buffers
    //



/*
    LOG(info) 
              << " n0 " << std::setw(3) << n0  
              << " n1 " << std::setw(3) << n1
              << " num_idx_add " << std::setw(3) <<  num_idx_add
              << " num_part_add " << std::setw(3) <<  num_part_add
              << " num_tran_add " << std::setw(3) << num_tran_add
              << " num_plan_add " << std::setw(3) << num_plan_add
              << " other_part_buffer  " << other_part_buffer->getShapeString()
              << " other_tran_buffer  " << other_tran_buffer->getShapeString()
              << " other_plan_buffer  " << other_plan_buffer->getShapeString()
              ;  
*/ 

  
}


/**
GParts::setContainingMaterial
-------------------------------

For flexibility persisted GParts should leave the outer containing material
set to a default marker name such as "CONTAINING_MATERIAL", 
to allow the GParts to be placed within other geometry

**/

void GParts::setContainingMaterial(const char* material)
{
    if(m_medium)
       LOG(fatal) << "setContainingMaterial called already " << m_medium 
       ;

    assert( m_medium == NULL && "GParts::setContainingMaterial WAS CALLED ALREADY " );
    m_medium = strdup(material); 

    unsigned field    = 0 ; 
    const char* from  = GParts::CONTAINING_MATERIAL ;
    const char* to    = material  ;
    const char* delim = "/" ;     

    m_bndspec->replaceField(field, from, to, delim );

    // all field zero *from* occurences are replaced with *to* 
}

void GParts::setSensorSurface(const char* surface)
{
    m_bndspec->replaceField(1, GParts::SENSOR_SURFACE, surface ) ; 
    m_bndspec->replaceField(2, GParts::SENSOR_SURFACE, surface ) ; 
}


void GParts::close()
{
    if(isClosed()) LOG(fatal) << "closed already " ;
    assert(!isClosed()); 
    m_closed = true ; 

    LOG(LEVEL) << "[" ; 
    registerBoundaries();

    if(!m_loaded)
    {
        makePrimBuffer(); 
    }
    
    dumpPrimBuffer(); 

    LOG(LEVEL) << "]" ; 

}

void GParts::registerBoundaries() // convert boundary spec names into integer codes using bndlib
{
   assert(m_bndlib); 
   unsigned int nbnd = m_bndspec->getNumKeys() ; 
   assert( getNumParts() == nbnd );

   if(m_verbosity > 0)
   LOG(LEVEL) 
         << " verbosity " << m_verbosity
         << " nbnd " << nbnd 
         << " NumParts " << getNumParts() 
         ;

   for(unsigned int i=0 ; i < nbnd ; i++)
   {
       const char* spec = m_bndspec->getKey(i);
       unsigned int boundary = m_bndlib->addBoundary(spec);
       setBoundary(i, boundary);

       if(m_verbosity > 1)
       LOG(LEVEL) 
             << " i " << std::setw(3) << i 
             << " " << std::setw(30) << spec
             << " --> "
             << std::setw(4) << boundary 
             << " " << std::setw(30) << m_bndlib->shortname(boundary)
             ;

   } 
}


/**
GParts::reconstructPartsPerPrim
---------------------------------

The "classic" partlist formed in python with opticks/ana/pmt/analytic.py  (pmt-ecd)
uses the nodeindex entry in the partlist buffer to identify which parts 
correspond to each solid eg PYREX,VACUUM,CATHODE,BOTTOM,DYNODE. 

Hence by counting parts keyed by the nodeIndex the below reconstructs 
the number of parts for each primitive.

In orther words "parts" are associated to their containing "prim" 
via the nodeIndex property.

For the CSG nodeTree things are simpler as each NCSG tree 
directly corresponds to a 1 GVolume and 1 GParts that
are added separtately, see GGeoTest::loadCSG.

**/

void GParts::reconstructPartsPerPrim()
{
    assert(isPartList());
    m_parts_per_prim.clear();

    unsigned int nmin(INT_MAX) ; 
    unsigned int nmax(0) ; 

    unsigned numParts = getNumParts() ; 

    LOG(info) 
        << " numParts " << numParts 
        ;
 
    // count parts for each nodeindex
    for(unsigned int i=0; i < numParts ; i++)
    {
        unsigned int nodeIndex = getNodeIndex(i);
        unsigned typ = getTypeCode(i);
        std::string  typName = CSG::Name((OpticksCSG_t)typ);
 
        LOG(info) 
            << " i " << std::setw(3) << i  
            << " nodeIndex " << std::setw(3) << nodeIndex
            << " typ " << std::setw(3) << typ 
            << " typName " << typName 
            ;  
                     
        m_parts_per_prim[nodeIndex] += 1 ; 

        if(nodeIndex < nmin) nmin = nodeIndex ; 
        if(nodeIndex > nmax) nmax = nodeIndex ; 
    }

    unsigned int num_prim = m_parts_per_prim.size() ;

    //assert(nmax - nmin == num_solids - 1);  // expect contiguous node indices
    if(nmax - nmin != num_prim - 1)
    {
        LOG(warning) 
            << " non-contiguous node indices"
            << " nmin " << nmin 
            << " nmax " << nmax
            << " num_prim " << num_prim
            << " part_per_add.size " << m_part_per_add.size()
            << " tran_per_add.size " << m_tran_per_add.size()
            ; 
    }
}





/**
GParts::makePrimBuffer
------------------------

Derives prim buffer from the part buffer

BUT it relies on the _per_add vectors so this 
only works from a live combined GParts instance, 
no one that has just been loaded from file.


Primbuffer acts as an "index" providing cross
referencing associating a primitive via
offsets to the parts/nodes, transforms and planes
relevant to the primitive.

prim/part/tran/plan buffers are used GPU side in cu/intersect_analytic.cu.


example for DYB mm5 (PMT assembly instance) 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

        ./GParts.txt : 41 
    ./partBuffer.npy : (41, 4, 4) 
    ./tranBuffer.npy : (12, 3, 4, 4) 
    ./primBuffer.npy : (5, 4) 
     ./idxBuffer.npy : (5, 4) 

    In [1]: pb   ## partBuffer.npy with (partOffset, numParts, tranOffset, planOffset)
    Out[1]: 
    array([[ 0, 15,  0,  0],
           [15, 15,  4,  0],
           [30,  7,  8,  0],
           [37,  3, 10,  0],
           [40,  1, 11,  0]], dtype=int32)
    //          ^^
    //         numParts are complete binary tree sizes


    In [1]: xb   ## idxBuffer.npy with (index,soIdx,lvIdx,height)  was added later for debugging only (so far)
    Out[1]: 
    array([[ 0, 54, 47,  3],
           [ 0, 55, 46,  3],
           [ 0, 56, 43,  2],
           [ 0, 57, 44,  1],
           [ 0, 58, 45,  0]], dtype=uint32)


**/

void GParts::makePrimBuffer()
{
    unsigned int num_prim = 0 ; 

    if(isPartList())
    {
        reconstructPartsPerPrim();
        num_prim = m_parts_per_prim.size() ;
    } 
    else if(isNodeTree() )
    {
        num_prim = m_part_per_add.size() ;
        assert( m_part_per_add.size() == num_prim );
        assert( m_tran_per_add.size() == num_prim );
        assert( m_plan_per_add.size() == num_prim );
    }
    else
    {
        assert(0);
    }


    LOG(LEVEL) 
        << " verbosity " << m_verbosity
        << " isPartList " << isPartList()
        << " isNodeTree " << isNodeTree()
        << " num_prim " << num_prim
        << " parts_per_prim.size " << m_parts_per_prim.size()
        << " part_per_add.size " << m_part_per_add.size()
        << " tran_per_add.size " << m_tran_per_add.size()
        << " plan_per_add.size " << m_plan_per_add.size()
        ; 

    nivec4* priminfo = new nivec4[num_prim] ;

    unsigned part_offset = 0 ; 
    unsigned tran_offset = 0 ; 
    unsigned plan_offset = 0 ; 

    if(isNodeTree())
    {
        unsigned n = 0 ; 
        for(unsigned i=0 ; i < num_prim ; i++)
        {
            unsigned tran_for_prim = m_tran_per_add[i] ; 
            unsigned plan_for_prim = m_plan_per_add[i] ; 
            unsigned parts_for_prim = m_part_per_add[i] ; 

            nivec4& pri = *(priminfo+n) ;

            pri.x = part_offset ; 
            pri.y = m_primflag == CSG_FLAGPARTLIST ? -parts_for_prim : parts_for_prim ;
            pri.z = tran_offset ; 
            pri.w = plan_offset ; 

            LOG(LEVEL) << "(nodeTree)priminfo " << pri.desc() ;       

            part_offset += parts_for_prim ; 
            tran_offset += tran_for_prim ; 
            plan_offset += plan_for_prim ; 

            n++ ; 
        }
    }
    else if(isPartList())
    {
        unsigned n = 0 ; 
        typedef std::map<unsigned int, unsigned int> UU ; 
        for(UU::const_iterator it=m_parts_per_prim.begin() ; it != m_parts_per_prim.end() ; it++)
        {
            //unsigned node_index = it->first ; 
            unsigned parts_for_prim = it->second ; 

            nivec4& pri = *(priminfo+n) ;

            pri.x = part_offset ; 
            pri.y = m_primflag == CSG_FLAGPARTLIST ? -parts_for_prim : parts_for_prim ;
            pri.z = 0 ; 
            pri.w = 0 ; 

            LOG(LEVEL) << "(partList) priminfo " << pri.desc() ;       

            part_offset += parts_for_prim ; 
            n++ ; 
        }
    }


    NPY<int>* buf = NPY<int>::make( num_prim, 4 );
    buf->setData((int*)priminfo);
    delete [] priminfo ; 

    setPrimBuffer(buf);
}



void GParts::dumpPrim(unsigned primIdx)
{
    // following access pattern of oxrap/cu/intersect_analytic.cu::intersect

    NPY<int>*    primBuffer = getPrimBuffer();
    NPY<float>*  partBuffer = getPartBuffer();
    //NPY<float>*  planBuffer = getPlanBuffer();

    if(!primBuffer) return ; 
    if(!partBuffer) return ; 

    glm::ivec4 prim = primBuffer->getQuadI(primIdx) ;

    int partOffset = prim.x ; 
    int numParts_   = prim.y ; 
    int tranOffset = prim.z ; 
    int planOffset = prim.w ; 

    unsigned numParts = abs(numParts_) ;
    unsigned primFlag = numParts_ < 0 ? CSG_FLAGPARTLIST : CSG_FLAGNODETREE  ; 

    unsigned num_zeros = 0 ; 
    unsigned num_nonzeros = 0 ; 

    for(unsigned p=0 ; p < numParts ; p++)
    {
        unsigned int partIdx = partOffset + p ;

        nquad q0, q1, q2, q3 ;

        q0.f = partBuffer->getVQuad(partIdx,0);  
        q1.f = partBuffer->getVQuad(partIdx,1);  
        q2.f = partBuffer->getVQuad(partIdx,2);  
        q3.f = partBuffer->getVQuad(partIdx,3);  

        unsigned typecode = q2.u.w ;
        assert(TYPECODE_J == 2 && TYPECODE_K == 3);

        bool iszero = typecode == 0 ; 
        if(iszero) num_zeros++ ; 
        else num_nonzeros++ ; 

        if(!iszero)
        LOG(info) 
            << " p " << std::setw(3) << p 
            << " partIdx " << std::setw(3) << partIdx
            << " typecode " << typecode
            << " CSG::Name " << CSG::Name((OpticksCSG_t)typecode)
            ;

    }

    LOG(info) 
        << " primIdx "    << std::setw(3) << primIdx 
        << " partOffset " << std::setw(3) << partOffset 
        << " tranOffset " << std::setw(3) << tranOffset 
        << " planOffset " << std::setw(3) << planOffset 
        << " numParts_ "  << std::setw(3) << numParts_
        << " numParts "   << std::setw(3) << numParts
        << " num_zeros "   << std::setw(5) << num_zeros
        << " num_nonzeros " << std::setw(5) << num_nonzeros
        << " primFlag "   << std::setw(5) << primFlag 
        << " CSG::Name "  << CSG::Name((OpticksCSG_t)primFlag) 
        << " prim "       << gformat(prim)
        ;
}


void GParts::dumpPrimBuffer(const char* msg)
{
    NPY<int>*    primBuffer = getPrimBuffer();
    NPY<float>*  partBuffer = getPartBuffer();
    if(!primBuffer) return ; 
    if(!partBuffer) return ; 


    unsigned num_prim = primBuffer->getNumItems() ; 
    unsigned num_part = partBuffer->getNumItems() ;  

    LOG(LEVEL) 
        << msg 
        << " verbosity " << m_verbosity
        << " num_prim " << num_prim 
        << " num_part " << num_part
        << " primBuffer " << primBuffer->getShapeString() 
        << " partBuffer " << partBuffer->getShapeString() 
        ;

    if( num_prim == 0 )
    {
        LOG(LEVEL) << " skip no prim " ; 
        return ; 
    }  


    assert( primBuffer->hasItemShape(4,0) && num_prim > 0  );
    assert( partBuffer->hasItemShape(4,4) && num_part > 0 );

    if(m_verbosity > 3)
    {
        for(unsigned primIdx=0 ; primIdx < num_prim ; primIdx++) dumpPrim(primIdx);
    }
}


void GParts::dumpPrimInfo(const char* msg, unsigned lim )
{
    unsigned numPrim = getNumPrim() ;
    unsigned ulim = std::min( numPrim, lim ) ; 

    LOG(info) 
        << msg 
        << " (part_offset, parts_for_prim, tran_offset, plan_offset) "
        << " numPrim: " << numPrim 
        << " ulim: " << ulim 
        ;

    for(unsigned i=0 ; i < numPrim ; i++)
    {
        if( ulim != 0 &&  i > ulim && i < numPrim - ulim ) continue ;    

        nivec4 pri = getPrimInfo(i);
        LOG(info) << pri.desc() ;
    }
}





void GParts::Summary(const char* msg, unsigned lim)
{
    LOG(info) 
        << msg 
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

    unsigned numParts = getNumParts() ;
    unsigned ulim = std::min( numParts, lim ) ; 

    for(unsigned i=0 ; i < numParts ; i++)
    {
        if( ulim != 0 &&  i > ulim && i < numParts - ulim ) continue ; 
   
        std::string bn = getBoundaryName(i);
        printf(" part %2u : node %2u type %2u boundary [%3u] %s  \n", i, getNodeIndex(i), getTypeCode(i), getBoundary(i), bn.c_str() ); 
    }
}



std::string GParts::desc()
{
    std::stringstream ss ; 
    ss 
       << " GParts "
       << " primflag " << std::setw(20) << getPrimFlagString()
       << " numParts " << std::setw(4) << getNumParts()
       << " numPrim " << std::setw(4) << getNumPrim()
       ;

    return ss.str(); 
}






unsigned GParts::getNumPrim() const 
{
    return m_prim_buffer ? m_prim_buffer->getShape(0) : 0 ; 
}
const char* GParts::getTypeName(unsigned partIdx) const 
{
    unsigned tc = getTypeCode(partIdx);
    return CSG::Name((OpticksCSG_t)tc);
}
std::string GParts::getTag(unsigned partIdx) const 
{
    unsigned tc = getTypeCode(partIdx);
    return CSG::Tag((OpticksCSG_t)tc);
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

nivec4 GParts::getPrimInfo(unsigned int iprim) const 
{
    const int* data = m_prim_buffer->getValues();
    const int* ptr = data + iprim*SK  ;   // SK is 4

    nivec4 pri = make_nivec4( *ptr, *(ptr+1), *(ptr+2), *(ptr+3) );
    return pri ;  
}


int GParts::getPartOffset(unsigned primIdx) const
{
    nivec4 primInfo = getPrimInfo(primIdx);
    return primInfo.x ;  
}
int GParts::getNumParts(unsigned primIdx) const 
{
    nivec4 primInfo = getPrimInfo(primIdx);
    return primInfo.y ;  
}
int GParts::getTranOffset(unsigned primIdx) const 
{
    nivec4 primInfo = getPrimInfo(primIdx);
    return primInfo.z ;  
}
int GParts::getPlanOffset(unsigned primIdx) const 
{
    nivec4 primInfo = getPrimInfo(primIdx);
    return primInfo.w ;  
}



/**
GParts::getBBox
----------------

Suspect this is only valid for old partlist which carry the bbox
rather than the CSG node tree which needs to call bounds methods
with the analytic parameters to get the bbox.

**/

nbbox GParts::getBBox(unsigned int i)
{
   gfloat3 min = getGfloat3(i, BBMIN_J, BBMIN_K );  
   gfloat3 max = getGfloat3(i, BBMAX_J, BBMAX_K );  
   nbbox bb = make_bbox(min.x, min.y, min.z, max.x, max.y, max.z);  
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

    LOG(debug) 
        << " part " << part 
        << " epsilon " << epsilon
        << " min " << gformat(min) 
        << " max " << gformat(max)
        ; 

}

const float* GParts::getPartValues(unsigned i, unsigned j, unsigned k) const 
{
    assert(i < getNumParts() );
    return m_part_buffer->getValuesConst(i, j, k);     
}

unsigned int GParts::getUInt(unsigned int i, unsigned int j, unsigned int k) const 
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




unsigned GParts::getNodeIndex(unsigned partIdx) const    
{
    return getUInt(partIdx, NODEINDEX_J, NODEINDEX_K);  // hmm NODEINDEX slot is used for GTRANSFORM 
}

unsigned GParts::getGTransform(unsigned partIdx) const 
{
    unsigned q3_u_w = getUInt(partIdx, GTRANSFORM_J, GTRANSFORM_K);
    return q3_u_w & 0x7fffffff ; 
}
bool GParts::getComplement(unsigned partIdx) const 
{
    unsigned q3_u_w = getUInt(partIdx, GTRANSFORM_J, GTRANSFORM_K);
    return q3_u_w & 0x80000000  ; 
}







unsigned GParts::getTypeCode(unsigned partIdx) const 
{
    return getUInt(partIdx, TYPECODE_J, TYPECODE_K);
}
unsigned GParts::getIndex(unsigned partIdx) const 
{
    return getUInt(partIdx, INDEX_J, INDEX_K);
}
unsigned GParts::getBoundary(unsigned partIdx) const 
{
    return getUInt(partIdx, BOUNDARY_J, BOUNDARY_K);
}





void GParts::setNodeIndex(unsigned int part, unsigned int nodeindex)
{
    setUInt(part, NODEINDEX_J, NODEINDEX_K, nodeindex);  // hmm NODEINDEX slot is used for GTRANSFORM 
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





void GParts::setBoundaryAll(unsigned int boundary)
{
    for(unsigned int i=0 ; i < getNumParts() ; i++) setBoundary(i, boundary);
}
void GParts::setNodeIndexAll(unsigned int nodeindex)
{
    for(unsigned int i=0 ; i < getNumParts() ; i++) setNodeIndex(i, nodeindex);
}


std::string GParts::getBoundaryName(unsigned partIdx) const 
{
    unsigned boundary = getBoundary(partIdx);
    std::string name = m_bndlib ? m_bndlib->shortname(boundary) : "" ;
    return name ;
}



void GParts::fulldump(const char* msg, unsigned lim)
{
    LOG(info) 
        << msg 
        << " lim " << lim 
        ; 

    dump(msg, lim);
    Summary(msg);

    NPY<float>* partBuf = getPartBuffer();
    NPY<int>*   primBuf = getPrimBuffer(); 

    partBuf->dump("partBuf");
    primBuf->dump("primBuf:partOffset/numParts/primIndex/0");
}

void GParts::dump(const char* msg, unsigned lim)
{
    LOG(info) 
        << msg
        << " lim " << lim 
        << " pbuf " << m_part_buffer->getShapeString()
        ; 

    dumpPrimInfo(msg, lim);

    NPY<float>* buf = m_part_buffer ; 
    assert(buf);
    assert(buf->getDimensions() == 3);

    unsigned ni = buf->getShape(0) ;
    unsigned nj = buf->getShape(1) ;
    unsigned nk = buf->getShape(2) ;

    unsigned ulim = std::min( ni, lim ) ; 

    LOG(info) 
        << " ni " << ni 
        << " lim " << lim
        << " ulim " << ulim
        ; 

    assert( nj == NJ );
    assert( nk == NK );

    for(unsigned i=0; i < ni; i++)
    {   
       if( ulim != 0 &&  i > ulim && i < ni - ulim ) continue ;    
       dumpPart(i); 
    }   
}

/**
GParts::dumpPart
-----------------

partIdx is absolute index of the part(aka CSG constituent node) amongst 
all prim within the composite GParts

**/

void GParts::dumpPart(unsigned partIdx)
{
    uif_t uif ; 

    unsigned tc = getTypeCode(partIdx);
    unsigned id = getIndex(partIdx);
    unsigned gtran = getGTransform(partIdx);
    unsigned bnd = getBoundary(partIdx);
    std::string  bn = getBoundaryName(partIdx);
    const char*  tn = getTypeName(partIdx);
    std::string csg = CSG::Name((OpticksCSG_t)tc);



    unsigned i = partIdx ; 
    NPY<float>* buf = m_part_buffer ; 
    float* data = buf->getValues();

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
                printf(" %6u <-INDEX   " , uif.u );
            }
            else if( j == BOUNDARY_J && k == BOUNDARY_K)
            {
                assert( uif.u == bnd );
                printf(" %6u <-bnd  ", uif.u );
            }
            else if( j == NODEINDEX_J && k == NODEINDEX_K)
                printf(" %10d (nodeIndex) ", uif.i );
            else
                printf(" %10.4f ", uif.f );
        }   
        printf("\n");
    }   
    printf("bn %s \n", bn.c_str() );

    if( gtran > 0 )
    {
        glm::mat4 t = getTran(gtran-1,0) ; 
        std::cout << gpresent( "t", t ) << std::endl ; 
        //glm::mat4 v = getTran(gtran-1,1) ; 
        //std::cout << gpresent( "v", v ) << std::endl ; 
        //glm::mat4 q = getTran(gtran-1,2) ; 
        //std::cout << gpresent( "q", q ) << std::endl ; 
    }
    else
    {
        std::cout << " gtran:" << gtran << std::endl ;  
    }

}



void GParts::dumpTran(const char* msg) const 
{
    plog::Severity level = DEBUG == 0 ? LEVEL : info ;  
    unsigned numTran = getNumTran(); 
    unsigned numParts = getNumParts(); 

    LOG(level) 
        <<  msg 
        << " numTran " << numTran
        << " numParts " << numParts
        ;

    for(int t=0 ; t < int(numTran) ; t++)
    {
        const glm::mat4& tr = getTran(t, 0); 
        LOG(level) 
            << std::setw(4)  << t
            << " " << GLMFormat::Format(tr) 
            ; 
    } 

    LOG(level) 
        << " numParts " << numParts
        ;

    for(unsigned p=0 ; p < numParts ; p++)
    {
        unsigned partIdx = p ; 
        unsigned gtran = getGTransform(partIdx);
        std::string tag = getTag(partIdx);
        std::stringstream ss ; 
        ss 
            << " partIdx " << std::setw(4) << partIdx  
            << " tag " << std::setw(3) << tag 
            << " gtran " << std::setw(4) << gtran 
            << " numTran " << std::setw(4) << numTran 
            ;

        if(gtran > 0  )
        {
             const glm::mat4& tr = getTran(gtran-1, 0); 
             ss << " " << GLMFormat::Format(tr)  ; 
        }
 
        std::string s = ss.str(); 
        LOG(level) << s ; 
    } 
}







/**
GParts::setVolumeIndex
-------------------------

For "global" bits of geometry (from mm0) its handy to keep a reference to 
the volume index at analytic lebel

NB this overwrites the NCSG tree index, one of the initial (1,4) 
NCSG identity indices (index,soIdx,lvIdx,height) coming in from 
the NCSG tree via GParts::Make

**/

const unsigned GParts::VOL_IDX = 0 ; 
const unsigned GParts::MESH_IDX = 1 ; 

void GParts::setVolumeIndex(unsigned idx)
{
    assert(0 && "do this in the Make : not after ctor") ; 

#ifdef GPARTS_DEBUG
    // only called once per node
    bool dupe = std::find(m_nix.begin(), m_nix.end(), idx ) != m_nix.end() ; 
    m_nix.push_back(idx);  
    if(dupe) LOG(fatal) << " dupe " << idx ; 
    assert(!dupe); 
#endif 

    assert( 1 == getNumIdx() ) ;  
    setUIntIdx( 0, VOL_IDX, idx ) ; 
}



unsigned GParts::getVolumeIndex(unsigned i) const
{
    return getUIntIdx(i, VOL_IDX ) ;  
}
unsigned GParts::getMeshIndex(unsigned i) const
{
    return getUIntIdx(i, MESH_IDX ) ;  
}


void GParts::setUIntIdx(unsigned i, unsigned j, unsigned idx)
{
    assert(1 == getNumIdx() && i == 0 );
    unsigned k=0u ; 
    unsigned l=0u ; 
    m_idx_buffer->setUInt(i,j,k,l, idx);
}
unsigned GParts::getUIntIdx(unsigned i, unsigned j ) const 
{
    assert( i < getNumIdx() ); 
    unsigned k=0u ; 
    unsigned l=0u ; 
    unsigned idx = m_idx_buffer->getUInt( i, j, k, l); 
    return idx ; 
}





template GGEO_API NPY<float>* GParts::LoadBuffer<float>(const char*, const char*) ;
template GGEO_API NPY<int>* GParts::LoadBuffer<int>(const char*, const char*) ;
template GGEO_API NPY<unsigned>* GParts::LoadBuffer<unsigned>(const char*, const char*) ;


