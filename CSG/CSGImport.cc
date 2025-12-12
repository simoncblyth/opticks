
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
    st->get_mmlabel( fd->mmlabel);    // populate fd->mmlabel from the stree
    st->get_meshname(fd->meshname);   // populate fd->meshname from the stree
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
        char ridx_type = st->get_ridx_type(ridx) ;
        switch(ridx_type)
        {
            case 'R': importSolidGlobal( ridx, ridx_type ) ; break ;   // remainder
            case 'T': importSolidGlobal( ridx, ridx_type ) ; break ;   // triangulate
            case 'F': importSolidFactor( ridx, ridx_type ) ; break ;   // factor
        }
    }
}

/**
CSGImport::importSolidRemainder_OLD : non-instanced global volumes
-------------------------------------------------------------------

cf::

    CSG_GGeo_Convert::convertSolid
    CSG_GGeo_Convert::convertPrim


AABB from each CSGPrim is combined using s_bb::IncludeAABB

Q: Is that assuming common frame for all CSGPrim in the CSGSolid ?
A: That assumption is true for prims of the remainder solid, but might not be for
   the prim of other solid.

**/
CSGSolid* CSGImport::importSolidRemainder_OLD(int ridx, const char* rlabel)
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
CSGImport::importSolidGlobal
-----------------------------

Generalizes the old CSGImport::importSolidRemainder to source the
stree level snode from either *rem* or *tri* depending on the
ridx_type returned by stree::get_ridx_type

**/

CSGSolid* CSGImport::importSolidGlobal(int ridx, char ridx_type )
{
    assert( ridx_type == 'R' || ridx_type == 'T' );  // remainder or triangulate

    std::string _rlabel = CSGSolid::MakeLabel(ridx_type,ridx) ;
    const char* rlabel = _rlabel.c_str();


    const std::vector<snode>* src = st->get_node_vector(ridx_type) ;
    assert( src );

    int num_src = src->size() ;

    LOG(LEVEL)
        << " ridx " << ridx
        << " ridx_type " << ridx_type
        << " rlabel " << rlabel
        << " num_src " << num_src
        ;

    std::array<float,6> bb = {} ;
    CSGSolid* so = fd->addSolid(num_src, rlabel);
    so->setIntent(ridx_type);

    for(int i=0 ; i < num_src ; i++)
    {
        int primIdx = i ;  // primIdx within the CSGSolid
        const snode& node = (*src)[primIdx] ;
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

CSGSolid* CSGImport::importSolidFactor(int ridx, char ridx_type )
{
    assert( ridx > 0 );
    assert( ridx_type == 'F' );

    std::string _rlabel = CSGSolid::MakeLabel(ridx_type,ridx) ;
    const char* rlabel = _rlabel.c_str();


    int  num_rem = st->get_num_remainder() ;
    assert( num_rem == 1 ) ;  // YEP: always one

    int num_factor = st->factor.size() ;
    assert( ridx - num_rem < num_factor );

    const sfactor& sf = st->factor[ridx-num_rem] ;
    int subtree = sf.subtree ;  // number of prim within the compound solid

    CSGSolid* so = fd->addSolid(subtree, rlabel);
    so->setIntent(ridx_type);

    int q_repeat_index = ridx ;
    int q_repeat_ordinal = 0 ;   // just first repeat

    std::vector<snode> nodes ;
    st->get_repeat_node(nodes, q_repeat_index, q_repeat_ordinal) ;

    LOG(LEVEL)
        << " ridx " << ridx
        << " ridx_type " << ridx_type
        << " num_rem " << num_rem
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


WIP:LISTNODE

* from binary tree point of view the listnode is just a leaf "prim"
* for a listnode only prim : its just one binary tree node with lots of child
* need smth similar to CSGMaker::makeList

1. get the binary tree nodes into complete binary tree vector (excluding the subs of any listnode)
2. count total subs of any listnodes TODO: move down into sn.h
3. addPrim to foundry with space for binary nodes and all subs



**/


CSGPrim* CSGImport::importPrim(int primIdx, const snode& node )
{
    int lvid = node.lvid ;
    const char* name = fd->getMeshName(lvid)  ;
    bool strip = true ;
    std::string soname = st->get_lvid_soname(lvid, strip);


    const sn* rt = sn::GetLVRoot(lvid);
    assert(rt);
    int idx_rc = rt->check_idx("CSGImport::importPrim.check_idx");


    // 1. get the binary tree nodes into complete binary tree vector (excluding the subs of any listnode)

    std::vector<const sn*> nds ;
    sn::GetLVNodesComplete(nds, lvid);   // many nullptr in unbalanced deep complete binary trees
    int bn = nds.size();                 // binary nodes

    // 2. count total subs for any listnodes of this lvid

    std::vector<const sn*> lns ;
    sn::GetLVListnodes( lns, lvid );
    int num_sub_total = sn::GetChildTotal( lns );

    int ln = lns.size();
    bool ln_expect = ln == 0 || ln == 1 ;


    bool dump_LVID = node.lvid == LVID || ln > 0 || idx_rc > 0 ;
    if(dump_LVID) std::cout
        << "[CSGImport::importPrim.dump_LVID:" << dump_LVID
        << " node.lvid " << node.lvid
        << " idx_rc " << idx_rc
        << " LVID " << LVID
        << " name " << ( name ? name : "-" )
        << " soname " << soname
        << " primIdx " << primIdx
        << " bn " << bn << "(binary nodes)"
        << " ln(subset of bn) " << ln
        << " ln_expect " << ( ln_expect ? "YES" : "NO " )
        << " num_sub_total " << num_sub_total
        << "\n"
        << "[rt.render\n"
        << rt->render()
        << "]rt.render\n"
        ;

    if(dump_LVID && ln > 0 ) std::cout
        << ".CSGImport::importPrim dumping as ln > 0 : solid contains listnode"
        << std::endl
        ;

    assert( ln_expect ); // simplify initial impl


    // 3. addPrim to foundry with space for binary nodes and all subs

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


    // HMM: would be simpler for listnode to contribute to the prim bb
    // if could add the listnode subs
    // immediately into their offset place after the binary tree
    // rather than collecting and adding later
    // BUT out of order node adding is not currently possible with CSGFoundry::addNode

    for(int i=0 ; i < bn ; i++)
    {
        int partIdx = i ;
        const sn* nd = nds[partIdx];

        CSGNode* n = nullptr ;
        if(nd && nd->is_listnode())
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
        if(!n->is_complemented_primitive()) s_bb::IncludeAABB( bb.data(), n->AABB(), out );
    }


    pr->setAABB( bb.data() );

    assert( root );

    // IsCompound : > CSG_ZERO, < CSG_LEAF
    //
    // Q: Is this actually needed by anything ?
    // A: YES, for example its how intersect_tree gets numNode,
    //    without the subNum would get no intersects onto booleans
    //
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

    if(dump_LVID) std::cout
        << "]CSGImport::importPrim.dump_LVID:" << dump_LVID
        << " node.lvid " << node.lvid
        << " LVID " << LVID
        << " name " << ( name ? name : "-" )
        << " soname " << soname
         << std::endl
        ;


    return pr ;
}









/**
CSGImport::importNode (cf CSG_GGeo_Convert::convertNode)
----------------------------------------------------------

Note similarity with the old CSG_GGeo_Convert::convertNode

An assert constrains the *sn* CSG constituent to be from the shape *lvid*
that is associated with the structural *snode*.

(snode)node
    structural node "parent", corresponding to Geant4 PV/LV
(sn)nd
    constituent CSG nd, corresponding to Geant4 G4VSolid

nodeIdx
    local 0-based index over the CSGNode that comprise the CSGPrim

(snode)node.index
    absolute structural node index with large values

(sn)nd.index
    csg level index

Lack of complement in snd.hh and inflexibility motivated the move to sn.h



**TODO: handling nodes where external bbox expected**

::

    if( expect_external_bbox )
    {
        assert(aabb);
        n->setAABB_Narrow( aabb );
    }

transform handling
~~~~~~~~~~~~~~~~~~~~

stree::get_combined_tran_and_aabb
   computes combined structural(snode) and CSG tree node(sn) transform
   and inplace applies that transform to th



**CSG Leaf/Tree Frame AABB ?**

The stree::get_combined_tran_and_aabb expects the sn.h AABB
to be leaf frame (not CSG tree frame needed by sn::uncoincide).

**Leaf CSGNode transforms**

For the instanced with node.repeat_index > 0,
transforms are within the instance frame.

For global remainder with node.repeat_index == 0, it will be the absolute transform
combining the CSG node transforms with the structural node transforms all the way down
from root.


**/

CSGNode* CSGImport::importNode(int nodeOffset, int partIdx, const snode& node, const sn* nd)
{
    if(nd) assert( node.lvid == nd->lvid );

    int  typecode = nd ? nd->typecode : CSG_ZERO ;
    bool leaf = CSG::IsLeaf(typecode) ;

    bool external_bbox_is_expected = CSG::ExpectExternalBBox(typecode);
    // CSG_CONVEXPOLYHEDRON, CSG_CONTIGUOUS, CSG_DISCONTIGUOUS, CSG_OVERLAP

    bool expect = external_bbox_is_expected == false ;
    LOG_IF(fatal, !expect)
        << " NOT EXPECTING LEAF WITH EXTERNAL BBOX EXPECTED "
        << " for node of type " << CSG::Name(typecode)
        << " nd.lvid " << ( nd ? nd->lvid : -1 )
        ;
    assert(expect);
    if(!expect) std::raise(SIGINT);

    std::array<double,6> bb ;
    double* aabb = leaf ? bb.data() : nullptr ;

    std::ostream* out = nullptr ;
    stree::VTR* t_stack = nullptr ;

    const Tran<double>* tv = leaf ? st->get_combined_tran_and_aabb( aabb, node, nd, out, t_stack ) : nullptr ;
    unsigned tranIdx = tv ?  1 + fd->addTran(tv) : 0 ;   // 1-based index referencing foundry transforms

    CSGNode* n = fd->addNode();
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

Invoked from CSGImport::import

The CSGFoundry calls should parallel CSG_GGeo_Convert::addInstances
the source is the stree instead of GGeo/GMergedMesh etc..

**/

void CSGImport::importInst()
{
    fd->addInstanceVector( st->inst_f4 );
}



