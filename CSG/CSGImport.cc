#include "scuda.h"
#include "squad.h"
#include "stran.h"
#include "OpticksCSG.h"

#include "stree.h"
#include "snd.hh"

#include "SSys.hh"
#include "SLOG.hh"

#include "CSGNode.h"
#include "CSGFoundry.h"
#include "CSGImport.h"

const plog::Severity CSGImport::LEVEL = SLOG::EnvLevel("CSGImport", "DEBUG" ); 


CSGImport::CSGImport( CSGFoundry* fd_ )
    :
    fd(fd_),
    st(nullptr)
{
}



/**
CSGImport::importTree
----------------------

Best guide for how to implement is cg CSG_GGeo_Convert

CSG_GGeo_Convert::convertSolid
   shows that need to start from top level of the compound solids 
   to fit in with the inflexible, having to declare everything ahead, 
   way of populating CSGFoundry 


Need to find stree.h equivalents to GGeo counterparts::

    In [9]: f.factor[:,:5]  ## eg number of compound solids from factor array ?
    Out[9]: 
    array([[         0,      25600,          0,          5,  929456433],
           [         1,      12615,          0,          7,  926363696],
           [         2,       4997,          0,          7,  825517361],
           [         3,       2400,          0,          6, 1715024176],
           [         4,        590,          0,          1,  825569379],
           [         5,        590,          0,          1,  825255221],
           [         6,        590,          0,          1,  946037859],
           [         7,        590,          0,          1,  946103859],
           [         8,        504,          0,        130, 1631151159]], dtype=int32)


**/

void CSGImport::importTree(const stree* st_)
{
    LOG(LEVEL) << "[" ;     
    assert( st == nullptr ); 
    st = st_ ; 
    assert(st); 

    importSolid();     

    LOG(LEVEL) << "]" ;     
}


/**
CSGImport::importSolid : "Solid" means compound Solid 
----------------------------------------------------------------

After CSG_GGeo_Convert::convertAllSolid

**/

void CSGImport::importSolid()
{
    for(int ridx=0 ; ridx < int(1+st->get_num_factor()) ; ridx++) 
    {
        std::string _rlabel = CSGSolid::MakeLabel('r',ridx) ;
        const char* rlabel = _rlabel.c_str(); 

        if( ridx == 0 )
        {
            importRemainderSolid(ridx, rlabel ); 
        }
        else
        {
            importFactorSolid(ridx, rlabel ); 
        }
    }
}
        
/**
CSGImport::importRemainderSolid : non-instanced global volumes 
--------------------------------------------------------------

cf::

    CSG_GGeo_Convert::convertSolid
    CSG_GGeo_Convert::convertPrim

**/
CSGSolid* CSGImport::importRemainderSolid(int ridx, const char* rlabel)
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
        const snode& nd = st->rem[i] ;

        CSGPrim* pr = importPrim( i, nd ) ;  
        LOG_IF( verbose, pr == nullptr) << " pr null " ;  
        //assert( pr );  

    }

    return so ; 
}




CSGSolid* CSGImport::importFactorSolid(int ridx, const char* rlabel)
{
    assert( ridx > 0 ); 

    int num_factor = st->factor.size() ; 
    assert( ridx - 1 < num_factor ); 

    const sfactor& sf = st->factor[ridx-1] ; 
    int subtree = sf.subtree ; 

    CSGSolid* so = fd->addSolid(subtree, rlabel); 

    int q_repeat_index = ridx ; 
    int q_repeat_ordinal = 0 ;   // just first repeat 

    std::vector<int> lvids ; 
    st->get_repeat_lvid(lvids, q_repeat_index, q_repeat_ordinal) ; 

    std::vector<snode> nodes ; 
    st->get_repeat_node(nodes, q_repeat_index, q_repeat_ordinal) ; 

    LOG(LEVEL) 
        << " ridx " << ridx 
        << " rlabel " << rlabel 
        << " num_factor " << num_factor
        << " lvids.size " << lvids.size() 
        << " nodes.size " << nodes.size() 
        << " subtree " << subtree
        ;

    assert( subtree == int(lvids.size()) ); 

    for(int i=0 ; i < subtree ; i++)
    {
        const snode& node = nodes[i] ;   // structural node
        int lvid = node.lvid ; 
        int lvid1 = lvids[i] ; 
        assert( lvid1 == lvid ); 

        CSGPrim* pr = importPrim( i, node );  
        LOG_IF( verbose, pr == nullptr) << " pr null " ;  
        //assert( pr ); 
    }

    return so ; 
}



CSGPrim* CSGImport::importPrim( int primIdx, const snode& node )  // structural node
{
    std::vector<snd> nds ; 
    snd::GetLVID( nds, node.lvid );     

    LOG(LEVEL) << " primIdx " << primIdx << " node.lvid " << node.lvid << " snd::GetLVID " << nds.size() ; 

    return nullptr ; 
}


/**
CSGImport::importPrim
----------------------

See::

    sysrap/tests/stree_load_test.sh 


HMM: not so simple the stree is raw n-ary tree it needs 
some preparation before turning into CSGPrim/CSGNode

NCSG::export_
    writes nodetree into transport buffers 

NCSG::export_tree_
NCSG::export_list_
NCSG::export_leaf_

NCSG::export_tree_list_prepare_
    explains subNum/subOffet in serialization 
    of trees with list nodes

nnode::find_list_nodes_r
nnode::is_list
    CSG::IsList(type)   CSG_CONTIGUOUS or CSG_DISCONTIGUOUS or CSG_OVERLAP      

nnode::subNum
nnode::subOffset

    CSG::IsCompound

CSGNode re:subNum subOffset
    Used by compound node types such as CSG_CONTIGUOUS, CSG_DISCONTIGUOUS and 
    the rootnode of boolean trees CSG_UNION/CSG_INTERSECTION/CSG_DIFFERENCE...
    Note that because subNum uses q0.u.x and subOffset used q0.u.y 
    this should not be used for leaf nodes. 

NCSG::export_tree_r
    assumes pure binary tree serializing to 2*idx+1 2*idx+2 


How to convert the general n-ary tree into binary-tree + subs layout ?
-------------------------------------------------------------------------

* root can be a list node or a leaf, ie zero complete binary tree nodes 

* so need to examine the n-ary tree looking for how much of it 
  is binary and pulling out non-binary nodes into list node subs 

* also depends on type : a compound with 2 subs is not treated as boolean


NOTICE HOW IT SHOULD BE EASIER NOW : AS ARE DOING THIS ALL IN 
ONE PLACE UNLIKE OLD IMPL WHICH IS RATHER SPREAD AROUND : npy,GGeo 

* THIS IS BECAUSE ARE NOW ABLE TO SERIALIZE n-ARY TREES : SO CAN DELAY 
  THE SWITCH UNTIL LATER 

* PERHAPS EVENTUALLY COULD NOT SWITCH AT ALL AND INTERSECT 
  AGAINST THE N-ARY TREE ON GPU (THATS FOR FAR FUTURE) 


Looking at scsg nodes with NumPy::
     
    st
    ./stree_load_test.sh ana

    In [9]: print(st.desc_csg(18))
    desc_csg lvid:18 st.f.soname[18]:GLw1.up10_up11_FlangeI_Web_FlangeII0x59f4850 
            ix   dp   sx   pt   nc   fc   sx   lv   tc   pm   bb   xf
    array([[ 32,   2,   0,  34,   0,  -1,  33,  18, 110,  25,  25,  -1],
           [ 33,   2,   1,  34,   0,  -1,  -1,  18, 110,  26,  26,   5],
           [ 34,   1,   0,  36,   2,  32,  35,  18,   1,  -1,  -1,  -1],
           [ 35,   1,   1,  36,   0,  -1,  -1,  18, 110,  27,  27,   6],
           [ 36,   0,  -1,  -1,   2,  34,  -1,  18,   1,  -1,  -1,  -1]], dtype=int32)

**/

