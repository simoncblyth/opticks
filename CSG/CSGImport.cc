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

void CSGImport::import()
{
    LOG(LEVEL) << "[" ;     

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

After CSG_GGeo_Convert::convertAllSolid

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
    int lvid = node.lvid ; 
    const char* name = fd->getMeshName(lvid)  ; 

    std::vector<const snd*> nds ; 
    snd::GetLVNodesComplete(nds, lvid);   // many nullptr in unbalanced deep complete binary trees
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
        const snd* nd = nds[i]; 
        importNode(i, node, nd ) ; 
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

Imported CSGNode currently missing the float param and bounding box::

    In [12]: a.node[0]
    Out[12]: 
    array([[120000., 120000., 120000.,      0.],
           [     0.,      0.,      0.,      0.],
           [-60000., -60000., -60000.,  60000.],
           [ 60000.,  60000.,      0.,      0.]], dtype=float32)

    In [13]: b.node[0]
    Out[13]: 
    array([[0., 0., 0., 0.],
           [0., 0., 0., 0.],
           [0., 0., 0., 0.],
           [0., 0., 0., 0.]], dtype=float32)



    In [14]: a.node[0].view(np.int32)
    Out[14]: 
    array([[1206542336, 1206542336, 1206542336,          0],
           [         0,          0,          0,          0],
           [-949329920, -949329920, -949329920, 1198153728],
           [1198153728, 1198153728,        110,          1]], dtype=int32)

    In [15]: b.node[0].view(np.int32)
    Out[15]: 
    array([[  0,   0,   0,   0],
           [  0,   0,   0,   0],
           [  0,   0,   0,   0],
           [  0,   0, 110,   1]], dtype=int32)

**/


CSGNode* CSGImport::importNode(int nodeIdx, const snode& node, const snd* nd)
{
    CSGNode cn = CSGNode::Zero() ; 
    if(nd)
    {
        assert( node.lvid == nd->lvid ); 
        const float* aabb = nullptr ;  
        const float* param6 = nullptr ;  
        // TODO: get param from snd narrowed to float (or narrow inside CSGNode) 

        cn = CSGNode::Make(nd->typecode, param6, aabb ) ;  
    }
    int typecode = cn.typecode() ; 

    const std::vector<float4>* pl = nullptr ;  // planes
    const Tran<double>* tv = nullptr ; 

    if( CSG::IsLeaf(typecode) )
    {    
        glm::tmat4x4<double> t(1.)  ; 
        glm::tmat4x4<double> v(1.) ; 

        st->get_combined_transform(t, v, node, nd, nullptr ); 

        tv = new Tran<double>(t, v);  
    }    

    unsigned tranIdx = tv ?  1 + fd->addTran(tv) : 0 ;   // 1-based index referencing foundry transforms

    CSGNode* n = fd->addNode(cn, pl);    // Tran gets narrowed

    n->setTransform(tranIdx);

    return n ; 
}

/**
CSGImport::importInst
---------------------------

The CSGFoundry calls should parallel CSG_GGeo_Convert::addInstances
the source is the stree instead of GGeo/GMergedMesh etc..

::

    In [9]: np.unique(np.where(a.inst != b.inst)[1])
    Out[9]: array([2, 3])

Previously (2,3) Off-by-one::

    np.all(a.inst[:,2,3].view(np.int32)==b.inst[:,2,3].view(np.int32))   : False
    np.all(a.inst[:,2,3].view(np.int32)==b.inst[:,2,3].view(np.int32)+1) : True


**/

void CSGImport::importInst()
{
    fd->addInstanceVector( st->inst_f4 ); 
}
