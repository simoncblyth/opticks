
#include <csignal>
#include "scuda.h"
#include "squad.h"
#include "stran.h"
#include "OpticksCSG.h"

#include "SSim.hh"
#include "stree.h"
#include "snd.hh"
#include "s_bb.h"

#include "ssys.h"
#include "SLOG.hh"

#include "CSGNode.h"
#include "CSGFoundry.h"
#include "CSGImport.h"

const plog::Severity CSGImport::LEVEL = SLOG::EnvLevel("CSGImport", "DEBUG" ); 

const int CSGImport::LVID = ssys::getenvint("LVID", -1); 
const int CSGImport::NDID = ssys::getenvint("NDID", -1); 


CSGImport::CSGImport( CSGFoundry* fd_ )
    :
    fd(fd_),
    st(nullptr)
{
    LOG_IF(fatal, fd == nullptr) << " fd(CSGFoundry) required " ; 
    assert( fd ) ; 
}



/**
CSGImport::import : populate CSGFoundry using geometry info from stree.h
--------------------------------------------------------------------------

Former equivalent from hybrid old+new morass wass CSG_GGeo_Convert

**/

void CSGImport::import()
{
    LOG(LEVEL) << "[" ;     

    st = fd->sim ? fd->sim->tree : nullptr ; 
    LOG_IF(fatal, st == nullptr) << " fd.sim(SSim) fd.st(stree) required " ; 
    assert(st); 


    importNames(); 
    importSolid();     
    importInst();     

    LOG(LEVEL) << "]" ;     
}


void CSGImport::importNames()
{
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

    std::array<float,6> bb = {} ; 
    CSGSolid* so = fd->addSolid(num_rem, rlabel); 

    for(int i=0 ; i < num_rem ; i++)
    {
        int primIdx = i ;  // primIdx within the CSGSolid
        const snode& node = st->rem[primIdx] ;
        CSGPrim* pr = importPrim( primIdx, node ) ;  
        assert( pr );  
        s_bb::IncludeAABB( bb.data(), pr->AABB() );  
    }
    s_bb::CenterExtent( &(so->center_extent.x), bb.data() ); 
    return so ; 
}



/**
CSGImport::importSolidFactor
-----------------------------

Note the simple way the bbox of each prim are combined to 
give the center_extent of the solid. This implies that 
the bbox of all the prim are from the same frame, which 
should be the frame of the outer prim of the instance. 
Otherwise the center extent would be incorrect, unless 
there is no relative transform differences between the
prims of the compound solid. 

TODO: confirm consistent frame for the prim bbox 

**/

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

    std::array<float,6> bb = {} ; 

    for(int i=0 ; i < subtree ; i++)
    {
        int primIdx = i ;  // primIdx within the CSGSolid
        const snode& node = nodes[primIdx] ;   // structural node

        CSGPrim* pr = importPrim( primIdx, node );  
        assert( pr ); 
        pr->setRepeatIdx(ridx); 

        s_bb::IncludeAABB( bb.data(), pr->AABB() );  
    }
    s_bb::CenterExtent( &(so->center_extent.x), bb.data() ); 

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


WIP: add listnode handling 

* from binary tree point of view the listnode is just a leaf "prim" 
* for a listnode only prim : its just one binary tree node with lots of child 

::

    (gdb) f 4
    #4  0x00007ffff2700d2f in sn::getLVNodesComplete (this=0x69ec50, nds=std::vector of length 344, capacity 344 = {...}) at /data/blyth/opticks_Debug/include/SysRap/sn.h:3620
    3620        assert( ns == 0 ); // CHECKING : AS IMPL LOOKS LIKE ONLY HANDLES BINARY NODES
    (gdb) p child.size()
    $10 = 343
    (gdb) 


For listnode handling see CSGMaker::makeList


**/


CSGPrim* CSGImport::importPrim(int primIdx, const snode& node ) 
{
    int lvid = node.lvid ; 
    const char* name = fd->getMeshName(lvid)  ; 

    std::vector<const sn*> nds ; 

    sn::GetLVNodesComplete(nds, lvid);   // many nullptr in unbalanced deep complete binary trees
    int bn = nds.size();     // binary nodes
    int ln = 0 ; 
    int num_sub_total = 0 ; 

    for(unsigned i=0 ; i < nds.size() ; i++)
    {
        const sn* nd = nds[i]; 
        if(nd->is_listnode())
        {
            ln += 1 ; 
            num_sub_total += nd->child.size() ;             
        }    
    }

    assert( ln == 0 || ln == 1 ); // simplify initial impl 


    bool dump_LVID = node.lvid == LVID || ln > 0  ; 
    if(dump_LVID) std::cout 
        << "CSGImport::importPrim"
        << " node.lvid " << node.lvid
        << " primIdx " << primIdx  
        << " bn " << bn  
        << " ln(subset of bn) " << ln 
        << " num_sub_total " << num_sub_total 
        << " dump_LVID " << dump_LVID  
        << std::endl 
        ; 


    CSGPrim* pr = fd->addPrim( bn + num_sub_total );

    pr->setMeshIdx(lvid);
    pr->setPrimIdx(primIdx);  // primIdx within the CSGSolid

    std::stringstream ss ; 
    std::ostream* out = dump_LVID ? &ss : nullptr  ; 

    std::array<float,6> bb = {} ; 

    CSGNode* root = nullptr ; 

    // for any listnode in the binary tree, collect referenced n-ary subs
    std::vector<const sn*> subs ;  

    int sub_offset = 0 ; 
    sub_offset += bn ; 

    for(int i=0 ; i < bn ; i++)
    {
        int partIdx = i ; 
        const sn* nd = nds[partIdx]; 

        CSGNode* n = nullptr ; 
        if(nd->is_listnode())
        {
            n = importListnode(pr->nodeOffset(), partIdx, node, nd ) ; 

            int num_sub = nd->child.size() ;
            for(int j=0 ; j < num_sub ; j++)
            {    
                const sn* c = nd->child[j]; 
                subs.push_back(c); 
            }
            n->setSubNum(num_sub);  
            n->setSubOffset(sub_offset); 
            sub_offset += num_sub ; 
        }
        else
        {
            n = importNode(pr->nodeOffset(), partIdx, node, nd ) ; 
        }
        assert(n); 
        if(root == nullptr) root = n ;   // first node becomes root 

        if(!n->is_complemented_primitive()) s_bb::IncludeAABB( bb.data(), n->AABB(), out ); 
    }

    assert( sub_offset == bn + num_sub_total ); 
    assert( int(subs.size()) == num_sub_total ); 

    for( int i=0 ; i < num_sub_total ; i++ )
    {
        const sn* nd = subs[i]; 
        CSGNode* n = importNode(pr->nodeOffset(), i, node, nd ); 
        assert( n ); 
    }


    pr->setAABB( bb.data() );

    assert( root ); 


    // IsCompound : > CSG_ZERO, < CSG_LEAF 
    // HMM: is this actually needed by anything 
    if(CSG::IsCompound(root->typecode()) && !CSG::IsList(root->typecode()))
    {
        assert( bn > 0 ); 
        root->setSubNum( bn ); 
        root->setSubOffset( 0 );   
    }


    LOG_IF(info, dump_LVID ) <<  ss.str() ; 
    LOG(LEVEL) 
        << " primIdx " << std::setw(4) << primIdx 
        << " lvid "    << std::setw(3) << lvid 
        << " binaryNodes(bn) "  << std::setw(3) << bn
        << " : " 
        << name 
        ; 

    return pr ; 
}









/**
CSGImport::importNode (cf CSG_GGeo_Convert::convertNode)
----------------------------------------------------------

Impl must stay close to CSG_GGeo_Convert::convertNode

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


Lack of complement in snd.hh and inflexibility motivated the move to sn.h 


**CSG Leaf/Tree Frame AABB ?**

The stree::get_combined_tran_and_aabb expects the sn.h AABB 
to be leaf frame (not CSG tree frame needed by sn::uncoincide).


**Leaf CSGNode transforms**

For the instanced with node.repeat_index > 0,  
transforms are within the instance frame.

For global remainder with node.repeat_index == 0, it will be the absolute transform 
combining the CSG node transforms with the structural node transforms all the way down 
from root. 



TODO : SUPPORT FOR CSGNode TYPES THAT NEED EXTERNAL BBOX::

    DEFERRED DOING UNTIL WANT TO TEST SOME EXAMPLES

    bool expect_external_bbox = CSG::ExpectExternalBBox(typecode); 
    // CSG_CONVEXPOLYHEDRON, CSG_CONTIGUOUS, CSG_DISCONTIGUOUS, CSG_OVERLAP

    LOG_IF(fatal, expect_external_bbox && !has_aabb )
        << " For node of type " << CSG::Name(typecode)
        << " nd.lvid " << nd->lvid 
        << " : EXPECT EXTERNAL AABB : BUT THERE IS NONE "
        ;

    if( expect_external_bbox )
    {   
        assert(has_aabb);   
        n->setAABB_Narrow( aabb ); 
    }


**/

CSGNode* CSGImport::importNode(int nodeOffset, int partIdx, const snode& node, const sn* nd)
{
    if(nd) assert( node.lvid == nd->lvid );

    int  typecode = nd ? nd->typecode : CSG_ZERO ; 
    bool leaf = CSG::IsLeaf(typecode) ; 

    bool external_bbox_is_expected = CSG::ExpectExternalBBox(typecode); 
    bool expect = external_bbox_is_expected == false ; 
    LOG_IF(fatal, !expect) << " NOT EXPECTING LEAF WITH EXTERNAL BBOX EXPECTED : DEFERRED UNTIL HAVE EXAMPLES " ; 
    assert(expect); 
    if(!expect) std::raise(SIGINT); 

    std::array<double,6> bb ; 
    double* aabb = leaf ? bb.data() : nullptr ;
    // NB : TRANSFORM VERY DEPENDENT ON node.repeat_index == 0 OR not 
    const Tran<double>* tv = leaf ? st->get_combined_tran_and_aabb( aabb, node, nd, nullptr ) : nullptr ; 
    unsigned tranIdx = tv ?  1 + fd->addTran(tv) : 0 ;   // 1-based index referencing foundry transforms

    CSGNode* n = fd->addNode();   
    //n->setIndex(nodeIdx);     // NOW AUTOMATED IN CSGFoundry::addNode
    n->setTypecode(typecode); 
    n->setBoundary(node.boundary); 
    n->setComplement( nd ? nd->complement : false ); 
    n->setTransform(tranIdx);
    n->setParam_Narrow( nd ? nd->getPA_data() : nullptr ); 
    n->setAABB_Narrow(aabb ? aabb : nullptr  ); 

    return n ; 
}

CSGNode* CSGImport::importListnode(int nodeOffset, int partIdx, const snode& node, const sn* nd)
{
    if(nd) assert( node.lvid == nd->lvid );
    assert( nd->is_listnode() ); 
    int typecode = nd->typecode ; 

    CSGNode* n = fd->addNode();   
    n->setTypecode(typecode); 
    n->setBoundary(node.boundary); 

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
