#include "scuda.h"
#include "squad.h"
#include "stran.h"
#include "OpticksCSG.h"

#include "SSim.hh"
#include "stree.h"
#include "snd.hh"

#include "SSys.hh"
#include "SLOG.hh"

#include "CSGNode.h"
#include "CSGFoundry.h"
#include "CSGImport.h"

const plog::Severity CSGImport::LEVEL = SLOG::EnvLevel("CSGImport", "DEBUG" ); 

const int CSGImport::LVID = SSys::getenvint("LVID", -1); 
const int CSGImport::NDID = SSys::getenvint("NDID", -1); 


CSGImport::CSGImport( CSGFoundry* fd_ )
    :
    fd(fd_),
    st(fd->sim ? fd->sim->tree : nullptr)
{
    assert( fd ) ; 
    assert( fd->sim ) ; 
    assert( fd->sim->tree ) ; 
    assert( st ); 
}



/**
CSGImport::import
------------------

Best guide for how to implement is cg CSG_GGeo_Convert

**/

void CSGImport::import()
{
    LOG(LEVEL) << "[" ;     

    importNames(); 
    importSolid();     
    importInst();     

    LOG(LEVEL) << "]" ;     
}


void CSGImport::importNames()
{
    assert(st); 
    st->get_mmlabel( fd->mmlabel);  
    st->get_meshname(fd->meshname);  
}


/**
CSGImport::importSolid : "Solid" means compound Solid 
----------------------------------------------------------------

Following prior CSG_GGeo_Convert::convertAllSolid

CSG_GGeo_Convert::convertSolid
   shows that need to start from top level of the compound solids 
   to fit in with the inflexible, having to declare everything ahead, 
   way of populating CSGFoundry 

**/

void CSGImport::importSolid()
{
    int num_ridx = st->get_num_ridx() ; 
    for(int ridx=0 ; ridx < num_ridx ; ridx++) 
    {
        std::string _rlabel = CSGSolid::MakeLabel('r',ridx) ;
        const char* rlabel = _rlabel.c_str(); 

        if( ridx == 0 )
        {
            importSolidRemainder(ridx, rlabel ); 
        }
        else
        {
            importSolidFactor(ridx, rlabel ); 
        }
    }
}
        
/**
CSGImport::importSolidRemainder : non-instanced global volumes 
--------------------------------------------------------------

cf::

    CSG_GGeo_Convert::convertSolid
    CSG_GGeo_Convert::convertPrim

**/
CSGSolid* CSGImport::importSolidRemainder(int ridx, const char* rlabel)
{
    assert( ridx == 0 ); 
    int num_rem = st->rem.size() ; 

    LOG(LEVEL) 
        << " ridx " << ridx 
        << " rlabel " << rlabel 
        << " num_rem " << num_rem 
        ; 

    CSGSolid* so = fd->addSolid(num_rem, rlabel); 

    for(int i=0 ; i < num_rem ; i++)
    {
        const snode& node = st->rem[i] ;
        CSGPrim* pr = importPrim( i, node ) ;  
        LOG_IF( verbose, pr == nullptr) << " pr null " ;  
        assert( pr );  
    }
    return so ; 
}




CSGSolid* CSGImport::importSolidFactor(int ridx, const char* rlabel)
{
    assert( ridx > 0 ); 

    int num_factor = st->factor.size() ; 
    assert( ridx - 1 < num_factor ); 

    const sfactor& sf = st->factor[ridx-1] ; 
    int subtree = sf.subtree ; 

    CSGSolid* so = fd->addSolid(subtree, rlabel); 

    int q_repeat_index = ridx ; 
    int q_repeat_ordinal = 0 ;   // just first repeat 

    std::vector<snode> nodes ; 
    st->get_repeat_node(nodes, q_repeat_index, q_repeat_ordinal) ; 

    LOG(LEVEL) 
        << " ridx " << ridx 
        << " rlabel " << rlabel 
        << " num_factor " << num_factor
        << " nodes.size " << nodes.size() 
        << " subtree " << subtree
        ;

    assert( subtree == int(nodes.size()) ); 

    for(int i=0 ; i < subtree ; i++)
    {
        const snode& node = nodes[i] ;   // structural node

        CSGPrim* pr = importPrim( i, node );  
        pr->setRepeatIdx(ridx); 

        LOG_IF( verbose, pr == nullptr) << " pr null " ;  
        assert( pr ); 
    }

    return so ; 
}




/**
CSGImport::importPrim
----------------------

Converting *snd/scsg* n-ary tree with compounds (eg multiunion and polycone) 
into the CSGNode serialized binary tree with list node constituents appended using 
subNum/subOffset referencing.   

* Despite the input *snd* tree being an n-ary tree (able to hold polycone and multiunion compounds)
  it must be traversed as a binary tree by regarding the compound nodes as effectively leaf node "primitives" 
  in order to generate the indices into the complete binary tree serialization in level order 

**/


CSGPrim* CSGImport::importPrim(int primIdx, const snode& node ) 
{
#ifdef WITH_SND
    CSGPrim* pr = importPrim_<snd>(primIdx, node ) ;  
#else
    CSGPrim* pr = importPrim_<sn>(primIdx, node ) ;  
#endif
    return pr ; 
} 


template<typename N>
CSGPrim* CSGImport::importPrim_(int primIdx, const snode& node ) 
{
    int lvid = node.lvid ; 
    const char* name = fd->getMeshName(lvid)  ; 

    std::vector<const N*> nds ; 

    assert(0 && "IMPL INCOMPLETE");
    //N::GetLVNodesComplete(nds, lvid);   // many nullptr in unbalanced deep complete binary trees
    int numParts = nds.size(); 


    bool dump_LVID = node.lvid == LVID ; 
    if(dump_LVID) std::cout 
        << "CSGImport::importPrim"
        << " node.lvid " << node.lvid
        << " primIdx " << primIdx  
        << " numParts " << numParts  
        << " dump_LVID " << dump_LVID  
        << std::endl 
        ; 


    CSGPrim* pr = fd->addPrim( numParts );

    pr->setMeshIdx(lvid);
    pr->setPrimIdx(primIdx);

    for(int i=0 ; i < numParts ; i++)
    {
        int partIdx = i ; 
        const N* nd = nds[partIdx]; 
        importNode<N>(pr->nodeOffset(), partIdx, node, nd ) ; 
    }

    LOG(LEVEL) 
        << " primIdx " << std::setw(4) << primIdx 
        << " lvid "    << std::setw(3) << lvid 
        << " numParts "  << std::setw(3) << numParts
        << " : " 
        << name 
        ; 

    return pr ; 
}



/**
CSGImport::importNode (cf CSG_GGeo_Convert::convertNode)
----------------------------------------------------------

An assert constrains the *snd* CSG constituent to be from the shape *lvid* 
that is associated with the structural *snode*. 

(snode)node
    structural node "parent", corresponding to Geant4 PV/LV 
(snd)nd 
    constituent CSG nd, corresponding to Geant4 G4VSolid 

nodeIdx
    local 0-based index over the CSGNode that comprise the CSGPrim

(snode)node.index
    absolute structural node index with large values 
     
(snd)nd.index
    csg level index


TODO: once get match tidy this up, looks like setting AABB twice 

**/


template<typename N>
CSGNode* CSGImport::importNode(int nodeOffset, int partIdx, const snode& node, const N* nd)
{
    CSGNode cn = CSGNode::Zero() ;
    const double* aabb = nullptr ;
    const double* param6 = nullptr ;
    bool complement = false ;

    aabb   = nd ? nd->getAABB() : nullptr ;
    param6 = nd ? nd->getParam() : nullptr ;
    complement = nd ? nd->complement : false ;
    // HMM: snd is missing complement, sn.h has it, U4Solid uses CSG_DIFFERENCE

    if(nd)
    {
        assert( node.lvid == nd->lvid );
        cn = CSGNode::MakeNarrow(nd->typecode, param6, aabb );
    }

    cn.setIndex(nodeOffset + partIdx) ;
    cn.setBoundary(node.boundary) ;
    cn.setComplement(complement) ;
    int typecode = cn.typecode() ;

    const std::vector<float4>* pl = nullptr ;  // planes
    const Tran<double>* tv = nullptr ;

    if( CSG::IsLeaf(typecode) )
    {
        glm::tmat4x4<double> t(1.) ;
        glm::tmat4x4<double> v(1.) ;

        assert(0 && "IMPL INCOMPLETE");
        //st->get_combined_transform(t, v, node, nd, nullptr );

        tv = new Tran<double>(t, v);
    }

    unsigned tranIdx = tv ?  1 + fd->addTran(tv) : 0 ;   // 1-based index referencing foundry transforms

    CSGNode* n = fd->addNode(cn, pl);    // Tran gets narrowed

    n->setTransform(tranIdx);


    bool expect_external_bbox = CSG::ExpectExternalBBox(typecode);

    // needs to follow CSG_GGeo_Convert::convertNode

    if( nd && nd->hasAABB() )
    {   
        /*
        LOG_IF(error, !expect_external_bbox) 
           << " For node of type " << CSG::Name(typecode)
           << " : DO NOT EXPECT AABB : BUT FIND ONE " 
           << " (maybe boolean tree for general sphere)" 
           ;
        */
        n->setAABB_Narrow( aabb ); 
    }
    else
    {
        LOG_IF(fatal, expect_external_bbox )
           << " For node of type " << CSG::Name(typecode)
           << " : EXPECT EXTERNAL AABB : BUT THERE IS NONE "
           ;
        if(expect_external_bbox) assert(0) ; 
        n->setAABBLocal(); // use params of each type to set the bbox 
    }

    const qat4* q = tranIdx > 0 ? fd->getTran(tranIdx-1u) : nullptr  ;
    if(q)  
    {
        q->transform_aabb_inplace( n->AABB() );
    }

    return n ; 
}

/**
CSGImport::importInst
---------------------------

The CSGFoundry calls should parallel CSG_GGeo_Convert::addInstances
the source is the stree instead of GGeo/GMergedMesh etc..

**/

void CSGImport::importInst()
{
    fd->addInstanceVector( st->inst_f4 ); 
}
