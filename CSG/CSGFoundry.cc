#include <iostream>
#include <iomanip>
#include <array>
#include <vector>
#include <set>
#include <algorithm>
#include <cstring>

#include <glm/glm.hpp>
#include <glm/gtx/string_cast.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "SSys.hh"
#include "SProc.hh"
#include "SStr.hh"
#include "SPath.hh"
#include "SBitSet.hh"
#include "SOpticksResource.hh"
#include "NP.hh"
#include "PLOG.hh"

#include "scuda.h"

#include "OpticksCSG.h"
#include "CSGSolid.h"
#include "CU.h"
#include "CSGFoundry.h"
#include "CSGName.h"
#include "CSGTarget.h"
#include "CSGGenstep.h"
#include "CSGMaker.h"
#include "CSGCopy.h"

const unsigned CSGFoundry::IMAX = 50000 ; 

const plog::Severity CSGFoundry::LEVEL = PLOG::EnvLevel("CSGFoundry", "DEBUG" ); 
const int CSGFoundry::VERBOSE = SSys::getenvint("VERBOSE", 0); 

std::string CSGFoundry::descComp() const 
{
    std::stringstream ss ; 
    ss << "CSGFoundry::descComp"
       << " bnd " << ( bnd ? bnd->sstr() : "-" ) 
       << " optical " << ( optical ? optical->sstr() : "-" ) 
       << " icdf " << ( icdf ? icdf->sstr() : "-" ) 
       ;  
    std::string s = ss.str(); 
    return s ; 
}

void CSGFoundry::setOpticalBnd(const NP* optical_, const NP* bnd_ )
{
    optical = optical_ ; 
    bnd = bnd_ ; 
    bd = bnd ? new CSGName(bnd->names) : nullptr  ; 

}

CSGFoundry::CSGFoundry()
    :
    d_prim(nullptr),
    d_node(nullptr),
    d_plan(nullptr),
    d_itra(nullptr),
    id(new CSGName(meshname)),
    target(new CSGTarget(this)),
    genstep(new CSGGenstep(this)),
    maker(new CSGMaker(this)),
    deepcopy_everynode_transform(true),
    last_added_solid(nullptr),
    last_added_prim(nullptr),
    optical(nullptr),
    bnd(nullptr),
    bd(nullptr),
    icdf(nullptr),
    meta(),
    fold(nullptr),
    cfbase(nullptr),
    geom(nullptr),
    loaddir(nullptr),
    origin(nullptr),
    elv(nullptr)
{
    init(); 
}

void CSGFoundry::init()
{
    // without sufficient reserved the vectors may reallocate on any push_back invalidating prior pointers 
    solid.reserve(IMAX); 
    prim.reserve(IMAX); 
    node.reserve(IMAX); 
    plan.reserve(IMAX); 
    tran.reserve(IMAX); 
    itra.reserve(IMAX); 
}


std::string CSGFoundry::desc() const 
{
    std::stringstream ss ; 
    ss << "CSGFoundry "
       << " num_total " << getNumSolidTotal()
       << " num_solid " << solid.size()
       << " num_prim " << prim.size()
       << " num_node " << node.size()
       << " num_plan " << plan.size()
       << " num_tran " << tran.size()
       << " num_itra " << itra.size()
       << " num_inst " << inst.size()
       << " ins " << ins.size()
       << " gas " << gas.size()
       << " ias " << ias.size()
       << " meshname " << meshname.size()
       << " mmlabel " << mmlabel.size()
       ;
    return ss.str(); 
}

std::string CSGFoundry::descSolid() const
{
    unsigned num_total = getNumSolidTotal(); 
    unsigned num_standard = getNumSolid(STANDARD_SOLID); 
    unsigned num_oneprim  = getNumSolid(ONE_PRIM_SOLID); 
    unsigned num_onenode  = getNumSolid(ONE_NODE_SOLID); 
    unsigned num_deepcopy = getNumSolid(DEEP_COPY_SOLID); 
    unsigned num_kludgebbox = getNumSolid(KLUDGE_BBOX_SOLID); 

    std::stringstream ss ; 
    ss << "CSGFoundry "
       << " total solids " << num_total
       << " STANDARD " << num_standard
       << " ONE_PRIM " << num_oneprim
       << " ONE_NODE " << num_onenode
       << " DEEP_COPY " << num_deepcopy
       << " KLUDGE_BBOX " << num_kludgebbox
       ;
    return ss.str(); 
}


std::string CSGFoundry::descMeshName() const 
{
    std::stringstream ss ; 

    ss << "CSGFoundry::descMeshName"
       << " meshname.size " << meshname.size()
       << std::endl ;
    for(unsigned i=0 ; i < meshname.size() ; i++)
        ss << std::setw(5) << i << " : " << meshname[i] << std::endl ;
    
    std::string s = ss.str(); 
    return s ; 
}


unsigned CSGFoundry::getNumMeshName() const 
{
    return meshname.size() ; 
}
unsigned CSGFoundry::getNumSolidLabel() const 
{
    return mmlabel.size() ; 
}
void CSGFoundry::CopyNames( CSGFoundry* dst, const CSGFoundry* src ) // static
{
    CopyMeshName( dst, src ); 
}

void CSGFoundry::CopyMeshName( CSGFoundry* dst, const CSGFoundry* src ) // static 
{
    assert( dst->meshname.size() == 0); 
    src->getMeshName(dst->meshname); 
    assert( src->meshname.size() == dst->meshname.size() );      
}

void CSGFoundry::getMeshName( std::vector<std::string>& mname ) const 
{
    for(unsigned i=0 ; i < meshname.size() ; i++)
    {
        const std::string& mn = meshname[i]; 
        mname.push_back(mn);   
    }
}

const std::string& CSGFoundry::getMeshName(unsigned midx) const 
{
    assert( midx < meshname.size() ); 
    return meshname[midx] ; 
}

const std::string& CSGFoundry::getBndName(unsigned bidx) const 
{
    assert( bnd ); 
    assert( bidx < bnd->names.size() ); 
    return bnd->names[bidx] ; 
}




const std::string CSGFoundry::descELV(const SBitSet* elv)
{
    std::vector<unsigned> include_pos ; 
    std::vector<unsigned> exclude_pos ; 
    elv->get_pos(include_pos, true ); 
    elv->get_pos(exclude_pos, false); 

    unsigned num_include = include_pos.size()  ; 
    unsigned num_exclude = exclude_pos.size()  ; 
    unsigned num_bits = elv->num_bits ; 
    assert( num_bits == num_include + num_exclude ); 

    std::stringstream ss ;  
    ss << "CSGFoundry::descELV" 
       << " elv.num_bits " << num_bits 
       << " include " << num_include
       << " exclude " << num_exclude
       << std::endl 
       ; 

    ss << "INCLUDE:" << include_pos.size() << std::endl << std::endl ;  
    for(unsigned i=0 ; i < include_pos.size() ; i++)
    {
        const unsigned& p = include_pos[i] ; 
        const std::string& mn = getMeshName(p) ; 
        ss << std::setw(3) << p << ":" << mn << std::endl ;  
    }

    ss << "EXCLUDE:" << exclude_pos.size() << std::endl << std::endl ;  
    for(unsigned i=0 ; i < exclude_pos.size() ; i++)
    {
        const unsigned& p = exclude_pos[i] ; 
        const std::string& mn = getMeshName(p) ; 
        ss << std::setw(3) << p << ":" << mn << std::endl ;  
    }

    std::string s = ss.str(); 
    return s ; 
} 



const std::string& CSGFoundry::getSolidLabel(unsigned sidx) const 
{
    assert( sidx < mmlabel.size() ); 
    return mmlabel[sidx] ; 
}

void CSGFoundry::addMeshName(const char* name) 
{
    meshname.push_back(name); 
}

void CSGFoundry::addSolidLabel(const char* label)
{
    mmlabel.push_back(label);  
}

int CSGFoundry::Compare( const CSGFoundry* a, const CSGFoundry* b )
{
    int mismatch = 0 ; 
    mismatch += CompareVec( "solid", a->solid, b->solid ); 
    mismatch += CompareVec( "prim" , a->prim , b->prim ); 
    mismatch += CompareVec( "node" , a->node , b->node ); 
    mismatch += CompareVec( "plan" , a->plan , b->plan ); 
    mismatch += CompareVec( "tran" , a->tran , b->tran ); 
    mismatch += CompareVec( "itra" , a->itra , b->itra ); 
    mismatch += CompareVec( "inst" , a->inst , b->inst ); 
    mismatch += CompareVec( "ins"  , a->ins , b->ins ); 
    mismatch += CompareVec( "gas"  , a->gas , b->gas ); 
    mismatch += CompareVec( "ias"  , a->ias , b->ias ); 
    if( mismatch != 0 ) LOG(fatal) << " mismatch FAIL ";  
    //assert( mismatch == 0 ); 
    return mismatch ; 
}


template<typename T>
int CSGFoundry::CompareVec( const char* name, const std::vector<T>& a, const std::vector<T>& b )
{
    int mismatch = 0 ; 

    bool size_match = a.size() == b.size() ; 
    if(!size_match) LOG(info) << name << " size_match FAIL " << a.size() << " vs " << b.size()    ; 
    if(!size_match) mismatch += 1 ; 
    if(!size_match) return mismatch ;  // below will likely crash if sizes are different 

    int data_match = memcmp( a.data(), b.data(), a.size()*sizeof(T) ) ; 
    if(data_match != 0) LOG(info) << name << " sizeof(T) " << sizeof(T) << " data_match FAIL " ; 
    if(data_match != 0) mismatch += 1 ; 

    int byte_match = CompareBytes( a.data(), b.data(), a.size()*sizeof(T) ) ;
    if(byte_match != 0) LOG(info) << name << " sizeof(T) " << sizeof(T) << " byte_match FAIL " ; 
    if(byte_match != 0) mismatch += 1 ; 

    if( mismatch != 0 ) LOG(fatal) << " mismatch FAIL for " << name ;  
    if( mismatch != 0 ) std::cout << " mismatch FAIL for " << name << std::endl ;  
    //assert( mismatch == 0 ); 
    return mismatch ; 
}

int CSGFoundry::CompareBytes(const void* a, const void* b, unsigned num_bytes)
{
    const char* ca = (const char*)a ; 
    const char* cb = (const char*)b ; 
    int mismatch = 0 ; 
    for(int i=0 ; i < int(num_bytes) ; i++ ) if( ca[i] != cb[i] ) mismatch += 1 ; 
    return mismatch ; 
}


template int CSGFoundry::CompareVec(const char*, const std::vector<CSGSolid>& a, const std::vector<CSGSolid>& b ) ; 
template int CSGFoundry::CompareVec(const char*, const std::vector<CSGPrim>& a, const std::vector<CSGPrim>& b ) ; 
template int CSGFoundry::CompareVec(const char*, const std::vector<CSGNode>& a, const std::vector<CSGNode>& b ) ; 
template int CSGFoundry::CompareVec(const char*, const std::vector<float4>& a, const std::vector<float4>& b ) ; 
template int CSGFoundry::CompareVec(const char*, const std::vector<qat4>& a, const std::vector<qat4>& b ) ; 
template int CSGFoundry::CompareVec(const char*, const std::vector<unsigned>& a, const std::vector<unsigned>& b ) ; 


void CSGFoundry::summary(const char* msg ) const 
{
    LOG(info) << msg << std::endl << descSolids() ; 
}

std::string CSGFoundry::descSolids() const 
{
    unsigned num_solids = getNumSolid(); 
    std::stringstream ss ; 
    ss 
        << "CSGFoundry::descSolids"
        << " num_solids " << num_solids 
        << std::endl
        ;

    for(unsigned i=0 ; i < num_solids ; i++)
    {
        const CSGSolid* so = getSolid(i); 
        ss << " " << so->desc() << std::endl ;
    }
    std::string s = ss.str(); 
    return s ; 
}





std::string CSGFoundry::descInst(unsigned ias_idx_, unsigned long long emm ) const
{
    std::stringstream ss ; 
    for(unsigned i=0 ; i < inst.size() ; i++)
    {
        const qat4& q = inst[i] ;      
        unsigned ins_idx, gas_idx, ias_idx ; 
        q.getIdentity(ins_idx, gas_idx, ias_idx);
        bool gas_enabled = emm == 0ull ? true : ( emm & (0x1ull << gas_idx)) ;  
        if( ias_idx_ == ias_idx && gas_enabled )
        {
            const CSGSolid* so = getSolid(gas_idx); 
            ss 
                << " ins " << std::setw(5) << ins_idx                
                << " gas " << std::setw(2) << gas_idx                
                << " ias " << std::setw(1) << ias_idx                
                << " so " << so->desc()
                << std::endl
                ;
        }
    }
    std::string s = ss.str(); 
    return s ; 
}







/**
CSGFoundry::iasBB
--------------------

bbox of the IAS obtained by transforming the center_extent cubes of all instances
hmm: could get a smaller bbox by using the bbox and not the ce of the instances 
need to add bb to solid...

**/

AABB CSGFoundry::iasBB(unsigned ias_idx_, unsigned long long emm ) const
{
    AABB bb = {} ;
    std::vector<float3> corners ; 
    for(unsigned i=0 ; i < inst.size() ; i++)
    {
        const qat4& q = inst[i] ;      
        unsigned ins_idx, gas_idx, ias_idx ; 
        q.getIdentity(ins_idx, gas_idx, ias_idx);
        bool gas_enabled = emm == 0ull ? true : ( emm & (0x1ull << gas_idx)) ;  
        if( ias_idx_ == ias_idx && gas_enabled )
        {
            const CSGSolid* so = getSolid(gas_idx); 
            corners.clear();       
            AABB::cube_corners(corners, so->center_extent);
            q.right_multiply_inplace( corners, 1.f );
            for(int i=0 ; i < int(corners.size()) ; i++) bb.include_point(corners[i]) ; 
        }
    }
    return bb ; 
}



/**
CSGFoundry::getMaxExtent
---------------------------

Kinda assumes the solids are all close to origin. This tends to 
work for a selection of one prim solids all from the same instance.  

**/

float CSGFoundry::getMaxExtent(const std::vector<unsigned>& solid_selection) const 
{
    float mxe = 0.f ; 
    for(unsigned i=0 ; i < solid_selection.size() ; i++)
    {   
        unsigned gas_idx = solid_selection[i] ; 
        const CSGSolid* so = getSolid(gas_idx); 
        float4 ce = so->center_extent ; 
        if(ce.w > mxe) mxe = ce.w ; 
        LOG(info) << " gas_idx " << std::setw(3) << gas_idx << " ce " << ce << " mxe " << mxe ; 
    }   
    return mxe ; 
}

std::string CSGFoundry::descSolids(const std::vector<unsigned>& solid_selection) const 
{
    std::stringstream ss ; 
    ss << "CSGFoundry::descSolids solid_selection " << solid_selection.size() << std::endl ; 
    for(unsigned i=0 ; i < solid_selection.size() ; i++)
    {   
        unsigned gas_idx = solid_selection[i] ; 
        const CSGSolid* so = getSolid(gas_idx); 
        //float4 ce = so->center_extent ; 
        //ss << " gas_idx " << std::setw(3) << gas_idx << " ce " << ce << std::endl ; 
        ss << so->desc() << std::endl ;  
    }   
    std::string s = ss.str(); 
    return s ; 
}

void CSGFoundry::gasCE(float4& ce, unsigned gas_idx ) const
{
    const CSGSolid* so = getSolid(gas_idx); 
    ce.x = so->center_extent.x ; 
    ce.y = so->center_extent.y ; 
    ce.z = so->center_extent.z ; 
    ce.w = so->center_extent.w ; 
}

void CSGFoundry::gasCE(float4& ce, const std::vector<unsigned>& gas_idxs ) const
{
    unsigned middle = gas_idxs.size()/2 ;  // target the middle selected solid : what about even ?
    unsigned gas_idx = gas_idxs[middle]; 

    const CSGSolid* so = getSolid(gas_idx); 
    ce.x = so->center_extent.x ; 
    ce.y = so->center_extent.y ; 
    ce.z = so->center_extent.z ; 
    ce.w = so->center_extent.w ; 
}







void CSGFoundry::iasCE(float4& ce, unsigned ias_idx_, unsigned long long emm ) const
{
    AABB bb = iasBB(ias_idx_, emm); 
    bb.center_extent(ce) ;
}

float4 CSGFoundry::iasCE(unsigned ias_idx_, unsigned long long emm ) const
{
    float4 ce = make_float4( 0.f, 0.f, 0.f, 0.f ); 
    iasCE(ce, ias_idx_, emm );
    return ce ; 
}


void CSGFoundry::dump() const 
{
    LOG(info) << "[" ; 
    dumpPrim(); 
    dumpNode(); 

    LOG(info) << "]" ; 
}

void CSGFoundry::dumpSolid() const 
{
    unsigned num_solid = getNumSolid(); 
    for(unsigned solidIdx=0 ; solidIdx < num_solid ; solidIdx++)
    {
        dumpSolid(solidIdx); 
    }
}

void CSGFoundry::dumpSolid(unsigned solidIdx) const 
{
    const CSGSolid* so = solid.data() + solidIdx ; 
    int primOffset = so->primOffset ; 
    int numPrim = so->numPrim  ; 
    
    std::cout 
        << " solidIdx " << std::setw(3) << solidIdx 
        << so->desc() 
        << " primOffset " << std::setw(5) << primOffset 
        << " numPrim " << std::setw(5) << numPrim 
        << std::endl
        ; 

    for(int primIdx=so->primOffset ; primIdx < primOffset + numPrim ; primIdx++)
    {
        const CSGPrim* pr = prim.data() + primIdx ; 
        int nodeOffset = pr->nodeOffset() ; 
        int numNode = pr->numNode() ;  

        std::cout
            << " primIdx " << std::setw(3) << primIdx << " "
            << pr->desc() 
            << " nodeOffset " << std::setw(4) << nodeOffset 
            << " numNode " << std::setw(4) << numNode
            << std::endl 
            ; 

        for(int nodeIdx=nodeOffset ; nodeIdx < nodeOffset + numNode ; nodeIdx++)
        {
            const CSGNode* nd = node.data() + nodeIdx ; 
            std::cout << nd->desc() << std::endl ; 
        }
    } 
}

int CSGFoundry::findSolidIdx(const char* label) const 
{
    int idx = -1 ; 
    if( label == nullptr ) return idx ; 
    for(unsigned i=0 ; i < solid.size() ; i++)
    {
        const CSGSolid& so = solid[i]; 
        if(strcmp(so.label, label) == 0) idx = i ;        
    }
    return idx ; 
}


/**
CSGFoundry::findSolidIdx
--------------------------

Find multiple idx with labels starting with the provided string, eg "r1", "r2", "r1p" or "r2p" 

This uses SStr:SimpleMatch which implements simple pattern matching with '$' 
indicating the terminator forcing exact entire match of what is prior to the '$'

**/

void CSGFoundry::findSolidIdx(std::vector<unsigned>& solid_idx, const char* label) const 
{
    if( label == nullptr ) return ; 

    std::vector<unsigned>& ss = solid_idx ; 

    std::vector<std::string> elem ; 
    SStr::Split(label, ',', elem ); 

    for(unsigned i=0 ; i < elem.size() ; i++)
    {
        const std::string& ele = elem[i] ; 
        for(unsigned j=0 ; j < solid.size() ; j++)
        {
            const CSGSolid& so = solid[j]; 

            bool match = SStr::SimpleMatch(so.label, ele.c_str()) ;
            unsigned count = std::count(ss.begin(), ss.end(), j ); 
            if(match && count == 0) ss.push_back(j) ; 
        }
    }

}

std::string CSGFoundry::descSolidIdx( const std::vector<unsigned>& solid_idx )
{
    std::stringstream ss ; 
    ss << "(" ; 
    for(int i=0 ; i < int(solid_idx.size()) ; i++) ss << solid_idx[i] << " " ; 
    ss << ")" ; 
    std::string s = ss.str() ; 
    return s ; 
}







void CSGFoundry::dumpPrim() const 
{
    std::string s = descPrim(); 
    LOG(info) << s ;
}

std::string CSGFoundry::descPrim() const 
{
    std::stringstream ss ; 
    for(unsigned idx=0 ; idx < solid.size() ; idx++) ss << descPrim(idx); 
    std::string s = ss.str(); 
    return s ; 
}

std::string CSGFoundry::descPrim(unsigned solidIdx) const 
{
    const CSGSolid* so = getSolid(solidIdx); 
    assert(so); 

    std::stringstream ss ; 
    ss << std::endl << so->desc() << std::endl ;  

    for(int primIdx=so->primOffset ; primIdx < so->primOffset+so->numPrim ; primIdx++)  
    {
        const CSGPrim* pr = getPrim(primIdx) ;  // note absolute primIdx
        assert(pr) ; 
        ss << "    primIdx " << std::setw(5) << primIdx << " : " << pr->desc() << std::endl ; 
    } 

    std::string s = ss.str(); 
    return s ; 
}

/**
CSGFoundry::detailPrim
------------------------

Used from CSGPrimTest 

**/

std::string CSGFoundry::detailPrim() const 
{
    std::stringstream ss ; 
    int numPrim = getNumPrim() ; 
    assert( int(prim.size()) == numPrim );  
    for(int primIdx=0 ; primIdx < std::min(10000, numPrim) ; primIdx++) ss << detailPrim(primIdx) << std::endl ;   
    std::string s = ss.str(); 
    return s ; 
}

/**
CSGFoundry::getPrimBoundary
----------------------------

Gets the boundary index of a prim. 
Currently this gets the boundary from all CSGNode of the 
prim and asserts that they are all the same. 

TODO: a faster version that just gets from the first node 

**/
int CSGFoundry::getPrimBoundary(unsigned primIdx) const
{
    std::set<unsigned> bnd ; 
    const CSGPrim* pr = getPrim(primIdx); 
    for(int nodeIdx=pr->nodeOffset() ; nodeIdx < pr->nodeOffset() + pr->numNode() ; nodeIdx++)
    {
        const CSGNode* nd = getNode(nodeIdx); 
        bnd.insert(nd->boundary()); 
    }
    assert( bnd.size() == 1 ); 
    int boundary = bnd.begin() == bnd.end() ? -1 : *bnd.begin() ; 
    return boundary ; 
}


/**
CSGFoundry::setPrimBoundary
---------------------------------------

Sets the boundary index for all CSGNode from the *primIdx* CSGPrim. 
This is intended for in memory changing of boundaries **within simple test geometries only**.

It would be unwise to apply this to full geometries and then persist the changed CSGFoundry
as that would be difficult to manage. 

With full geometries the boundaries are set during geometry 
translation in for example CSG_GGeo.

NB intersect identity is a combination of primIdx and instanceIdx so does not need to be set 

**/

void CSGFoundry::setPrimBoundary(unsigned primIdx, const char* bname ) 
{
    unsigned count = 0 ; 
    int bnd = bd->getIndex(bname, count ); 
    bool bname_found = count == 1 && bnd > -1  ;
    if(!bname_found) 
       LOG(fatal) 
          << " primIdx " << primIdx
          << " bname " << bname
          << " bnd " << bnd
          << " count " << count
          << " bname_found " << bname_found
          << " bd.getNumName " << bd->getNumName() 
          << " bd.detail " << std::endl 
          << bd->detail()
          ;

    assert( bname_found ); 
    unsigned boundary = bnd ; 

    setPrimBoundary(primIdx, boundary); 
}

void CSGFoundry::setPrimBoundary(unsigned primIdx, unsigned boundary ) 
{
    const CSGPrim* pr = getPrim(primIdx); 
    assert( pr ); 
    for(int nodeIdx=pr->nodeOffset() ; nodeIdx < pr->nodeOffset() + pr->numNode() ; nodeIdx++)
    {
        CSGNode* nd = getNode_(nodeIdx); 
        nd->setBoundary(boundary); 
    }
}





std::string CSGFoundry::detailPrim(unsigned primIdx) const 
{
    const CSGPrim* pr = getPrim(primIdx); 
    unsigned gasIdx = pr->repeatIdx(); 
    unsigned meshIdx = pr->meshIdx(); 
    unsigned pr_primIdx = pr->primIdx(); 
    const char* meshName = id->getName(meshIdx);

    int numNode = pr->numNode() ; 
    int nodeOffset = pr->nodeOffset() ; 
    int boundary = getPrimBoundary(primIdx); 
    const char* bndName = bd->getName(boundary);

    float4 ce = pr->ce(); 

    std::stringstream ss ; 
    ss  
        << std::setw(10) << SStr::Format(" pri:%d", primIdx )
        << std::setw(10) << SStr::Format(" lpr:%d", pr_primIdx )
        << std::setw(8)  << SStr::Format(" gas:%d", gasIdx )
        << std::setw(8)  << SStr::Format(" msh:%d", meshIdx)
        << std::setw(8)  << SStr::Format(" bnd:%d", boundary) 
        << std::setw(8)  << SStr::Format(" nno:%d", numNode )
        << std::setw(10)  << SStr::Format(" nod:%d", nodeOffset )
        << " ce " 
        << "(" << std::setw(10) << std::fixed << std::setprecision(2) << ce.x 
        << "," << std::setw(10) << std::fixed << std::setprecision(2) << ce.y
        << "," << std::setw(10) << std::fixed << std::setprecision(2) << ce.z
        << "," << std::setw(10) << std::fixed << std::setprecision(2) << ce.w
        << ")" 
        << " meshName " << std::setw(15) << ( meshName ? meshName : "-" )
        << " bndName "  << std::setw(15) << ( bndName  ? bndName  : "-" )
        ;   

    std::string s = ss.str();
    return s ; 
}




std::string CSGFoundry::descPrimSpec() const 
{
    unsigned num_solids = getNumSolid(); 
    std::stringstream ss ; 
    ss 
        << "CSGFoundry::descPrimSpec"
        << " num_solids " << num_solids 
        << std::endl
        ;

    for(unsigned i=0 ; i < num_solids ; i++) ss << descPrimSpec(i) << std::endl ;
 
    std::string s = ss.str(); 
    return s ; 
}

std::string CSGFoundry::descPrimSpec(unsigned solidIdx) const 
{
    unsigned gas_idx = solidIdx ; 
    CSGPrimSpec ps = getPrimSpec(gas_idx);
    return ps.desc() ; 
}






void CSGFoundry::dumpPrim(unsigned solidIdx) const 
{
    std::string s = descPrim(solidIdx); 
    LOG(info) << std::endl << s ;
}


void CSGFoundry::getNodePlanes(std::vector<float4>& planes, const CSGNode* nd) const 
{
    unsigned tc = nd->typecode(); 
    bool has_planes = CSG::HasPlanes(tc) ; 
    if(has_planes)
    {
        for(unsigned planIdx=nd->planeIdx() ; planIdx < nd->planeIdx() + nd->planeNum() ; planIdx++)
        {  
            const float4* pl = getPlan(planIdx);  
            planes.push_back(*pl); 
        }
    }
}


/**
CSGFoundry::getSolidPrim
----------------------------

Use *solidIdx* to get CSGSolid pointer *so* and then use 
the *so->primOffset* together with *primIdxRel* to get the CSGPrim pointer. 

**/

const CSGPrim*  CSGFoundry::getSolidPrim(unsigned solidIdx, unsigned primIdxRel) const 
{
    const CSGSolid* so = getSolid(solidIdx); 
    assert(so); 

    unsigned primIdx = so->primOffset + primIdxRel ; 
    const CSGPrim* pr = getPrim(primIdx); 
    assert(pr); 

    return pr ; 
}








void CSGFoundry::dumpNode() const
{
    LOG(info) << std::endl << descNode(); 
} 

void CSGFoundry::dumpNode(unsigned solidIdx) const
{
    LOG(info) << std::endl << descNode(solidIdx); 
} 

std::string CSGFoundry::descNode() const 
{
    std::stringstream ss ;
    for(unsigned idx=0 ; idx < solid.size() ; idx++) ss << descNode(idx) << std::endl ; 
    std::string s = ss.str(); 
    return s ; 
}

std::string CSGFoundry::descNode(unsigned solidIdx) const 
{
    const CSGSolid* so = solid.data() + solidIdx ; 
    //const CSGPrim* pr0 = prim.data() + so->primOffset ; 
    //const CSGNode* nd0 = node.data() + pr0->nodeOffset() ;  

    std::stringstream ss ;
    ss << std::endl << so->desc() << std::endl  ;

    for(int primIdx=so->primOffset ; primIdx < so->primOffset+so->numPrim ; primIdx++)
    {
        const CSGPrim* pr = prim.data() + primIdx ; 
        int numNode = pr->numNode() ; 
        for(int nodeIdx=pr->nodeOffset() ; nodeIdx < pr->nodeOffset()+numNode ; nodeIdx++)
        {
            const CSGNode* nd = node.data() + nodeIdx ; 
            ss << "    nodeIdx " << std::setw(5) << nodeIdx << " : " << nd->desc() << std::endl ; 
        }
    } 

    std::string s = ss.str(); 
    return s ; 
}

std::string CSGFoundry::descTran(unsigned solidIdx) const 
{
    const CSGSolid* so = solid.data() + solidIdx ; 
    //const CSGPrim* pr0 = prim.data() + so->primOffset ; 
    //const CSGNode* nd0 = node.data() + pr0->nodeOffset() ;  

    std::stringstream ss ;
    ss << std::endl << so->desc() << std::endl  ;

    for(int primIdx=so->primOffset ; primIdx < so->primOffset+so->numPrim ; primIdx++)
    {
        const CSGPrim* pr = prim.data() + primIdx ; 
        int numNode = pr->numNode() ; 
        for(int nodeIdx=pr->nodeOffset() ; nodeIdx < pr->nodeOffset()+numNode ; nodeIdx++)
        {
            const CSGNode* nd = node.data() + nodeIdx ; 
            unsigned tranIdx = nd->gtransformIdx(); 
            
            const qat4* tr = tranIdx > 0 ? getTran(tranIdx-1) : nullptr ;  
            const qat4* it = tranIdx > 0 ? getItra(tranIdx-1) : nullptr ;  
            ss << "    tranIdx " << std::setw(5) << tranIdx << " : " << ( tr ? tr->desc('t') : "" ) << std::endl ; 
            ss << "    tranIdx " << std::setw(5) << tranIdx << " : " << ( it ? it->desc('i') : "" ) << std::endl ; 
        }
    } 
    std::string s = ss.str(); 
    return s ; 
}



const CSGNode* CSGFoundry::getSolidPrimNode(unsigned solidIdx, unsigned primIdxRel, unsigned nodeIdxRel) const 
{
    const CSGPrim* pr = getSolidPrim(solidIdx, primIdxRel); 
    assert(pr); 
    unsigned nodeIdx = pr->nodeOffset() + nodeIdxRel ; 
    const CSGNode* nd = getNode(nodeIdx); 
    assert(nd); 
    return nd ; 
}   



/**
CSGFoundry::getPrimSpec
----------------------

Provides the specification to access the AABB and sbtIndexOffset of all CSGPrim 
of a CSGSolid.  The specification includes pointers, counts and stride.

NB PrimAABB is distinct from NodeAABB. Cannot directly use NodeAABB 
because the number of nodes for each prim (node tree) varies meaning 
that the strides are irregular. 

Prim Selection
~~~~~~~~~~~~~~~~

HMM: Prim selection will also require new primOffset for all solids, 
so best to implement it by spawning a new CSGFoundry with the selection applied.
Then the CSGFoundry code can stay the same just with different solid and prim 
and applying the selection can be focussed into one static method. 

HMM: but its not all of CSGFoundry that needs to have selection 
applied its just the solid and prim. Could prune nodes and transforms 
too, but probably not worthwhile. 

How to implement ? Kinda like CSG_GGeo translation but starting 
from another instance of CSGFoundry.

Also probably better to do enabledmergedmesh solid selection this
way too rather than smearing ok->isEnabledMergedMesh all over CSGOptiX/SBT
Better for SBT creation not to be mixed up with geometry selection.

**/

CSGPrimSpec CSGFoundry::getPrimSpec(unsigned solidIdx) const 
{
    CSGPrimSpec ps = d_prim ? getPrimSpecDevice(solidIdx) : getPrimSpecHost(solidIdx) ; 
    if(ps.device == false) LOG(info) << "WARNING using host PrimSpec, upload first " ; 
    return ps ; 
}
CSGPrimSpec CSGFoundry::getPrimSpecHost(unsigned solidIdx) const 
{
    const CSGSolid* so = solid.data() + solidIdx ; 
    CSGPrimSpec ps = CSGPrim::MakeSpec( prim.data(),  so->primOffset, so->numPrim ); ; 
    ps.device = false ; 
    return ps ; 
}
CSGPrimSpec CSGFoundry::getPrimSpecDevice(unsigned solidIdx) const 
{
    assert( d_prim ); 
    const CSGSolid* so = solid.data() + solidIdx ;  // get the primOffset from CPU side solid
    CSGPrimSpec ps = CSGPrim::MakeSpec( d_prim,  so->primOffset, so->numPrim ); ; 
    ps.device = true ; 
    return ps ; 
}

void CSGFoundry::checkPrimSpec(unsigned solidIdx) const 
{
    CSGPrimSpec ps = getPrimSpec(solidIdx);  
    LOG(info) << "[ solidIdx  " << solidIdx ; 
    ps.downloadDump(); 
    LOG(info) << "] solidIdx " << solidIdx ; 
}

void CSGFoundry::checkPrimSpec() const 
{
    for(unsigned solidIdx = 0 ; solidIdx < getNumSolid() ; solidIdx++)
    {
        checkPrimSpec(solidIdx); 
    }
}



unsigned CSGFoundry::getNumSolid(int type_) const 
{ 
    unsigned count = 0 ; 
    for(unsigned i=0 ; i < solid.size() ; i++)
    {
        const CSGSolid& so = solid[i] ; 
        if(so.type == type_ ) count += 1 ;  
    } 
    return count ; 
} 

unsigned CSGFoundry::getNumSolid() const {  return getNumSolid(STANDARD_SOLID); } 
unsigned CSGFoundry::getNumSolidTotal() const { return solid.size(); } 



unsigned CSGFoundry::getNumPrim() const  { return prim.size();  } 
unsigned CSGFoundry::getNumNode() const  { return node.size(); }
unsigned CSGFoundry::getNumPlan() const  { return plan.size(); }
unsigned CSGFoundry::getNumTran() const  { return tran.size(); }
unsigned CSGFoundry::getNumItra() const  { return itra.size(); }
unsigned CSGFoundry::getNumInst() const  { return inst.size(); }

const CSGSolid*  CSGFoundry::getSolid(unsigned solidIdx) const { return solidIdx < solid.size() ? solid.data() + solidIdx  : nullptr ; } 
const CSGPrim*   CSGFoundry::getPrim(unsigned primIdx)   const { return primIdx  < prim.size()  ? prim.data()  + primIdx  : nullptr ; } 
const CSGNode*   CSGFoundry::getNode(unsigned nodeIdx)   const { return nodeIdx  < node.size()  ? node.data()  + nodeIdx  : nullptr ; }  
CSGNode*         CSGFoundry::getNode_(unsigned nodeIdx)        { return nodeIdx  < node.size()  ? node.data()  + nodeIdx  : nullptr ; }  

const float4*    CSGFoundry::getPlan(unsigned planIdx)   const { return planIdx  < plan.size()  ? plan.data()  + planIdx  : nullptr ; }
const qat4*      CSGFoundry::getTran(unsigned tranIdx)   const { return tranIdx  < tran.size()  ? tran.data()  + tranIdx  : nullptr ; }
const qat4*      CSGFoundry::getItra(unsigned itraIdx)   const { return itraIdx  < itra.size()  ? itra.data()  + itraIdx  : nullptr ; }
const qat4*      CSGFoundry::getInst(unsigned instIdx)   const { return instIdx  < inst.size()  ? inst.data()  + instIdx  : nullptr ; }






const CSGSolid*  CSGFoundry::getSolid_(int solidIdx_) const { 
    unsigned solidIdx = solidIdx_ < 0 ? unsigned(solid.size() + solidIdx_) : unsigned(solidIdx_)  ;   // -ve counts from end
    return getSolid(solidIdx); 
}   

const CSGSolid* CSGFoundry::getSolidByName(const char* name) const  // caution stored labels truncated to 4 char 
{
    unsigned missing = ~0u ; 
    unsigned idx = missing ; 
    for(unsigned i=0 ; i < solid.size() ; i++) if(strcmp(solid[i].label, name) == 0) idx = i ;  
    assert( idx != missing ); 
    return getSolid(idx) ; 
}

/**
CSGFoundry::getSolidIdx
----------------------

Without sufficient reserve allocation this is unreliable as pointers go stale on reallocations.

**/

unsigned CSGFoundry::getSolidIdx(const CSGSolid* so) const 
{
    unsigned idx = ~0u ; 
    for(unsigned i=0 ; i < solid.size() ; i++) 
    {
       const CSGSolid* s = solid.data() + i ; 
       LOG(info) << " i " << i << " s " << s << " so " << so ; 
       if(s == so) idx = i ;  
    } 
    assert( idx != ~0u ); 
    return idx ; 
}




void CSGFoundry::makeDemoSolids()
{
    maker->makeDemoSolids(); 
}


CSGSolid* CSGFoundry::make(const char* name)
{
    return maker->make(name); 
}



/**
CSGFoundry::addNode
--------------------

Note that the planeIdx and planeNum of the CSGNode are 
rewritten based on the number of planes for this nd 
and the number of planes collected already into
the global plan vector. 

**/

CSGNode* CSGFoundry::addNode(CSGNode nd, const std::vector<float4>* pl, const Tran<double>* tr  )
{
    if(!last_added_prim) LOG(fatal) << "must addPrim prior to addNode" ; 
    assert( last_added_prim ); 

    unsigned globalNodeIdx = node.size() ;  

    unsigned nodeOffset = last_added_prim->nodeOffset(); 
    unsigned numNode = last_added_prim->numNode(); 
    unsigned localNodeIdx = globalNodeIdx - nodeOffset ; 

    bool ok_localNodeIdx = localNodeIdx < numNode ; 
    if(!ok_localNodeIdx) LOG(fatal) 
        << " TOO MANY addNode FOR Prim " 
        << " localNodeIdx " << localNodeIdx
        << " numNode " << numNode
        << " globalNodeIdx " << globalNodeIdx
        << " (must addNode only up to the declared numNode from the addPrim call) "
        ;
    assert( ok_localNodeIdx  ); 

    bool ok_globalNodeIdx = globalNodeIdx < IMAX  ;
    if(!ok_globalNodeIdx)
    {
        LOG(fatal) 
            << " FATAL : OUT OF RANGE "
            << " globalNodeIdx " << globalNodeIdx 
            << " IMAX " << IMAX
            ;
    }

    assert( ok_globalNodeIdx ); 

    unsigned num_planes = pl ? pl->size() : 0 ; 
    if(num_planes > 0)
    {
        nd.setTypecode(CSG_CONVEXPOLYHEDRON) ; 
        nd.setPlaneIdx(plan.size());    
        nd.setPlaneNum(num_planes);    
        for(unsigned i=0 ; i < num_planes ; i++) addPlan((*pl)[i]);  
    }

    if(tr)
    {
        unsigned trIdx = 1u + addTran(tr);  // 1-based idx, 0 meaning None
        nd.setTransform(trIdx);  
    }

    node.push_back(nd); 
    last_added_node = node.data() + globalNodeIdx ;
    return last_added_node ; 
}




/**
CSGFoundry::addNodes
----------------------

Pointer to the last added node is returned

**/

CSGNode* CSGFoundry::addNodes(const std::vector<CSGNode>& nds )
{
    unsigned idx = node.size() ; 
    for(unsigned i=0 ; i < nds.size() ; i++) 
    {
        const CSGNode& nd = nds[i]; 
        idx = node.size() ;     // number of nodes prior to adding this one 
        assert( idx < IMAX ); 
        node.push_back(nd); 
    }
    return node.data() + idx ; 
}

CSGNode*  CSGFoundry::addNode(AABB& bb, CSGNode nd )
{
    CSGNode* n = addNode(nd); 
    bb.include_aabb( n->AABB() ); 
    return n ; 
}

CSGNode* CSGFoundry::addNodes(AABB& bb, std::vector<CSGNode>& nds, const std::vector<const Tran<double>*>* trs  )
{
    if( trs == nullptr ) return addNodes(nds); 
    
    unsigned num_nd = nds.size() ; 
    unsigned num_tr = trs ? trs->size() : 0  ; 
    if( num_tr > 0 ) assert( num_nd == num_tr );

    CSGNode* n = nullptr ;  
    for(unsigned i=0 ; i < num_nd ; i++) 
    {
        CSGNode& nd = nds[i]; 
        const Tran<double>* tr = trs ? (*trs)[i] : nullptr ; 
        n = addNode(nd); 
        if(tr)
        {
            bool transform_node_aabb = true ; 
            addNodeTran(n, tr, transform_node_aabb ); 
        }
        bb.include_aabb( n->AABB() ); 
    }
    return n ; 
}









float4* CSGFoundry::addPlan(const float4& pl )
{
    unsigned idx = plan.size(); 
    assert( idx < IMAX ); 
    plan.push_back(pl); 
    return plan.data() + idx ; 
}



/**
CSGFoundry::addTran
---------------------

When tr argument is nullptr an identity transform is added.

**/
 
template<typename T>
unsigned CSGFoundry::addTran( const Tran<T>* tr  )
{
   return tr == nullptr ? addTran() : addTran_(tr); 
}
template unsigned CSGFoundry::addTran<float>(const Tran<float>* ) ;
template unsigned CSGFoundry::addTran<double>(const Tran<double>* ) ;


 
template<typename T>
unsigned CSGFoundry::addTran_( const Tran<T>* tr  )
{
    qat4 t(glm::value_ptr(tr->t));  // narrowing when T=double
    qat4 v(glm::value_ptr(tr->v)); 
    unsigned idx = addTran(&t, &v); 
    return idx ; 
}

template unsigned CSGFoundry::addTran_<float>(const Tran<float>* ) ;
template unsigned CSGFoundry::addTran_<double>(const Tran<double>* ) ;

unsigned CSGFoundry::addTran( const qat4* tr, const qat4* it )
{
    unsigned idx = tran.size();   // size before push_back 
    assert( tran.size() == itra.size()) ; 
    tran.push_back(*tr); 
    itra.push_back(*it); 
    return idx ;  
}

/**
CSGFoundry::addTran
----------------------

Add identity transform to tran and itra arrays and return index. 

**/
unsigned CSGFoundry::addTran()
{
    qat4 t ; 
    t.init(); 
    qat4 v ; 
    v.init(); 
    unsigned idx = addTran(&t, &v); 
    return idx ; 
}

/**
CSGFoundry::addTranPlaceholder
-------------------------------

Adds transforms tran and itra if none have yet been added 

**/
void CSGFoundry::addTranPlaceholder()
{
    unsigned idx = tran.size();   // size before push_back 
    assert( tran.size() == itra.size()) ; 
    if( idx == 0 )
    {
        addTran();   
    }
}



/**
CSGFoundry::addNodeTran
------------------------

Adds tranform and associates it to the CSGNode

**/

template<typename T>
const qat4* CSGFoundry::addNodeTran( CSGNode* nd, const Tran<T>* tr, bool transform_node_aabb  )
{
    unsigned transform_idx = 1 + addTran(tr);      // 1-based idx, 0 meaning None
    nd->setTransform(transform_idx); 
    const qat4* q = getTran(transform_idx-1u) ;   // storage uses 0-based 

    if( transform_node_aabb )
    {
        q->transform_aabb_inplace( nd->AABB() );  
    }
    return q ; 
}


template const qat4* CSGFoundry::addNodeTran<float>(  CSGNode* nd, const Tran<float>* , bool ) ;
template const qat4* CSGFoundry::addNodeTran<double>( CSGNode* nd, const Tran<double>*, bool ) ;


void CSGFoundry::addNodeTran(CSGNode* nd )
{
    unsigned transform_idx = 1 + addTran();      // 1-based idx, 0 meaning None
    nd->setTransform(transform_idx); 
}
 







/**
CSGFoundry::addInstance
------------------------

Used for example from CSG_GGeo_Convert::addInstances

**/

void CSGFoundry::addInstance(const float* tr16, unsigned gas_idx, unsigned ias_idx )
{
    qat4 instance(tr16) ;  // identity matrix if tr16 is nullptr 
    unsigned ins_idx = inst.size() ;

    instance.setIdentity( ins_idx, gas_idx, ias_idx );

    LOG(LEVEL) 
        << " ins_idx " << ins_idx 
        << " gas_idx " << gas_idx 
        << " ias_idx " << ias_idx 
        ;

    inst.push_back( instance );
} 

void CSGFoundry::addInstancePlaceholder()
{
    const float* tr16 = nullptr ; 
    unsigned gas_idx = 0 ; 
    unsigned ias_idx = 0 ; 

    addInstance(tr16, gas_idx, ias_idx );  
}






/**
CSGFoundry::addPrim
---------------------

Offsets counts for  node, tran and plan are 
persisted into the CSGPrim. 
Thus must addPrim prior to adding any node, 
tran or plan needed for that prim.

The nodeOffset_ argument default of -1 signifies
to set the nodeOffset of the Prim to the count of
preexisting Prim.  This is appropriate when are 
adding new nodes.  

When reusing preexisting nodes, provide a nodeOffset_ argument > -1 

**/

CSGPrim* CSGFoundry::addPrim(int num_node, int nodeOffset_ )  
{
    if(!last_added_solid) LOG(fatal) << "must addSolid prior to addPrim" ; 
    assert( last_added_solid ); 

    unsigned primOffset = last_added_solid->primOffset ; 
    unsigned numPrim = last_added_solid->numPrim ; 

    unsigned globalPrimIdx = prim.size(); 
    unsigned localPrimIdx = globalPrimIdx - primOffset ; 

    bool in_range = localPrimIdx < numPrim ; 
    if(!in_range) LOG(fatal) 
        << " TOO MANY addPrim FOR SOLID " 
        << " localPrimIdx " << localPrimIdx
        << " numPrim " << numPrim
        << " globalPrimIdx " << globalPrimIdx
        << " (must addPrim only up to to the declared numPrim from the addSolid call) "
        ;
           
    assert( in_range ); 

    int nodeOffset = nodeOffset_ < 0 ? int(node.size()) : nodeOffset_ ; 

    CSGPrim pr = {} ;

    pr.setNumNode(num_node) ; 
    pr.setNodeOffset(nodeOffset); 
    //pr.setSbtIndexOffset(globalPrimIdx) ;  // <--- bug : needs to be local 
    pr.setSbtIndexOffset(localPrimIdx) ; 

    pr.setMeshIdx(-1) ;                // metadata, that may be set by caller 

    pr.setTranOffset(tran.size());     // HMM are tranOffset and planOffset used now that use global referencing  ?
    pr.setPlanOffset(plan.size());     // but still handy to keep them for debugging 

    assert( globalPrimIdx < IMAX ); 
    prim.push_back(pr); 

    last_added_prim = prim.data() + globalPrimIdx ;
    last_added_node = nullptr ; 

    return last_added_prim  ; 
}


/**
CSGFoundry::getMeshPrimCopies
------------------------------

Collect Prims with the supplied mesh_idx 

**/
void CSGFoundry::getMeshPrimCopies(std::vector<CSGPrim>& select_prim, unsigned mesh_idx ) const 
{
    CSGPrim::select_prim_mesh(prim, select_prim, mesh_idx); 
}

void CSGFoundry::getMeshPrimPointers(std::vector<const CSGPrim*>& select_prim, unsigned mesh_idx ) const 
{
    CSGPrim::select_prim_pointers_mesh(prim, select_prim, mesh_idx); 
}

/**
CSGFoundry::getMeshPrim
------------------------

Selects prim pointers that match the *midx* mesh index
and then return the ordinal-th one of them. 

midx
    mesh index
mord
    mesh ordinal 

**/

const CSGPrim* CSGFoundry::getMeshPrim( unsigned midx, unsigned mord ) const 
{
    std::vector<const CSGPrim*> select_prim ; 
    getMeshPrimPointers(select_prim, midx );     

    bool mord_in_range = mord < select_prim.size() ; 
    if(!mord_in_range) 
    {   
        LOG(error)  << " midx " << midx << " mord " << mord << " select_prim.size " << select_prim.size() << " mord_in_range " << mord_in_range ;   
        return nullptr ; 
    }   

    const CSGPrim* pr = select_prim[mord] ; 
    return pr ; 
}

/**
CSGFoundry::getNumMeshPrim
-----------------------------

Returns the number of prim with the *mesh_idx* in entire geometry. 
Using CSGPrim::count_prim_mesh

The MOI mesh-ordinal values must always be less than the number of mesh prim.  

**/
unsigned CSGFoundry::getNumMeshPrim(unsigned mesh_idx ) const 
{
    return CSGPrim::count_prim_mesh(prim, mesh_idx); 
}

unsigned CSGFoundry::getNumSelectedPrimInSolid(const CSGSolid* solid, const SBitSet* elv ) const 
{
    unsigned num_selected_prim = 0 ;      
    for(int primIdx=solid->primOffset ; primIdx < solid->primOffset+solid->numPrim ; primIdx++)
    {    
        const CSGPrim* pr = getPrim(primIdx); 
        unsigned meshIdx = pr->meshIdx() ; 
        bool selected = elv == nullptr ? true : elv->is_set(meshIdx) ; 
        num_selected_prim += int(selected) ;  
    }
    return num_selected_prim ; 
}


/**
CSGFoundry::descMeshPrim
--------------------------

Presents table:

+----------+------------------+---------------+
| midx     |   numMeshPrim    |   meshName    |
+----------+------------------+---------------+


midx
   mesh index corresponding to lvIdx

numMeshPrim
   number of prim in entire geometry with this midx

meshName
   name coming from the source geometry


Notice that the meshName might not be unique, it is 
merely obtained from the source geometry solid name. 
In this case meshName addressing is not very useful
and it is necessary to address using the midx.  

**/

std::string CSGFoundry::descMeshPrim() const 
{
    std::stringstream ss ; 
    unsigned numName = id->getNumName(); 
    ss 
        << "CSGFoundry::descMeshPrim  id.numName " << numName << std::endl  
        << std::setw(4) << "midx"
        << " "
        << std::setw(12) << "numMeshPrim"
        << " "
        << "meshName"
        << std::endl
        ;

    for(unsigned midx=0 ; midx < numName ; midx++)
    {
        const char* meshName = id->getName(midx); 
        unsigned numMeshPrim = getNumMeshPrim(midx); 
        ss 
            << std::setw(4) << midx 
            << " "
            << std::setw(12) << numMeshPrim 
            << " "
            << meshName
            << std::endl 
            ;
    }
    return ss.str(); 
}




/**
CSGFoundry::addSolid
----------------------

The Prim offset is persisted into the CSGSolid
thus must addSolid prior to adding any prim
for the solid. 

The default primOffset_ argument of -1 signifies are about to 
add fresh Prim and need to obtain the primOffset for the added solid 
from the number of prim that have been collected previously.

Using a primOffset_ > -1 indicates that the added solid is reusing 
already existing Prim (eg for debugging) and the primOffset should be
set from this argument.

**/

CSGSolid* CSGFoundry::addSolid(unsigned numPrim, const char* label, int primOffset_ )
{
    unsigned idx = solid.size(); 

    assert( idx < IMAX ); 

    int primOffset = primOffset_ < 0 ? prim.size() : primOffset_ ;

    CSGSolid so = CSGSolid::Make( label, numPrim , primOffset ); 

    solid.push_back(so); 

    last_added_solid = solid.data() + idx  ;  // retain last_added_solid for getting the solid local primIdx 
    last_added_prim = nullptr ; 
    last_added_node = nullptr ; 
 
    return last_added_solid ;  
}



/**
CSGFoundry::addDeepCopySolid
-------------------------------

Used only from CSG_GGeo_Convert::addDeepCopySolid


TODO: will probably want to always add transforms as the point of making 
deep copies is to allow making experimental changes to the copies 
eg for applying progressive shrink scaling to check whether problems are caused 
by bbox being too close to each other


 
**/
CSGSolid* CSGFoundry::addDeepCopySolid(unsigned solidIdx, const char* label )
{
    std::string cso_label = label ? label : CSGSolid::MakeLabel('d', solidIdx) ; 

    LOG(info) << " cso_label " << cso_label ; 
    std::cout << " cso_label " << cso_label << std::endl ; 

    const CSGSolid* oso = getSolid(solidIdx); 
    unsigned numPrim = oso->numPrim ; 

    AABB solid_bb = {} ; 
    CSGSolid* cso = addSolid(numPrim, cso_label.c_str()); 
    cso->type = DEEP_COPY_SOLID ; 

    for(int primIdx=oso->primOffset ; primIdx < oso->primOffset+oso->numPrim ; primIdx++)
    {
        const CSGPrim* opr = prim.data() + primIdx ; 

        unsigned numNode = opr->numNode()  ; 
        int nodeOffset_ = -1 ; // as deep copying, -1 declares that will immediately add numNode new nodes

        AABB prim_bb = {} ; 
        CSGPrim* cpr = addPrim(numNode, nodeOffset_ );

        cpr->setMeshIdx(opr->meshIdx());    // copy the metadata that makes sense to be copied 
        cpr->setRepeatIdx(opr->repeatIdx()); 
        cpr->setPrimIdx(opr->primIdx()); 

        for(int nodeIdx=opr->nodeOffset() ; nodeIdx < opr->nodeOffset()+opr->numNode() ; nodeIdx++)
        {
            const CSGNode* ond = node.data() + nodeIdx ; 
            unsigned o_tranIdx = ond->gtransformIdx(); 

            CSGNode cnd = {} ; 
            CSGNode::Copy(cnd, *ond );   // straight copy reusing the transform reference  

            const qat4* tra = nullptr ; 
            const qat4* itr = nullptr ; 
            unsigned c_tranIdx = 0u ; 
 
            if( o_tranIdx > 0 )
            {
                tra = getTran(o_tranIdx-1u) ; 
                itr = getItra(o_tranIdx-1u) ; 
            }
            else if( deepcopy_everynode_transform )
            {
                tra = new qat4 ; 
                itr = new qat4 ; 
            }

            if( tra && itr )
            {
                c_tranIdx = 1 + addTran(tra, itr);  // add fresh transforms, as this is deep copy  
                std::cout 
                    << " o_tranIdx " << o_tranIdx 
                    << " c_tranIdx " << c_tranIdx 
                    << " deepcopy_everynode_transform " << deepcopy_everynode_transform
                    << std::endl
                    ; 
                std::cout << " tra  " << tra->desc('t') << std::endl ; 
                std::cout << " itr  " << itr->desc('i') << std::endl ; 
            } 


            // TODO: fix this in CSGNode 
            bool c0 = cnd.is_complement(); 
            //cnd.zeroTransformComplement(); 
            //cnd.setComplement(c0) ; 
            //cnd.setTransform( c_tranIdx );   
            cnd.setTransformComplement(c_tranIdx, c0);  

            unsigned c_tranIdx2 = cnd.gtransformIdx() ; 

            bool match = c_tranIdx2 == c_tranIdx ;  
            if(!match) std::cout << "set/get transform fail c_tranIdx " << c_tranIdx << " c_tranIdx2 " << c_tranIdx2 << std::endl ; 
            assert(match); 


            cnd.setAABBLocal() ;  // reset to local with no transform applied
            if(tra)  
            {
                tra->transform_aabb_inplace( cnd.AABB() ); 
            }
            prim_bb.include_aabb( cnd.AABB() ); 
            addNode(cnd);       
        }                  // over nodes of the Prim 
         
        cpr->setAABB(prim_bb.data()); 
        solid_bb.include_aabb(prim_bb.data()) ; 
    }    // over Prim of the Solid

    cso->center_extent = solid_bb.center_extent() ;  
    return cso ; 
}


void CSGFoundry::DumpAABB(const char* msg, const float* aabb) // static 
{
    int w = 4 ; 
    LOG(info) << msg << " " ; 
    LOG(info) << " | " ; 
    for(int l=0 ; l < 3 ; l++) LOG(info) << std::setw(w) << *(aabb+l) << " " ; 
    LOG(info) << " | " ; 
    for(int l=0 ; l < 3 ; l++) LOG(info) << std::setw(w) << *(aabb+l+3) << " " ; 
    LOG(info) << " | " ; 
    for(int l=0 ; l < 3 ; l++) LOG(info) << std::setw(w) << *(aabb+l+3) - *(aabb+l)  << " " ; 
    LOG(info) ; 
}



#ifdef __APPLE__
const char* CSGFoundry::BASE = "$TMP/GeoChain_Darwin" ;
#else
const char* CSGFoundry::BASE = "$TMP/GeoChain" ;
#endif

const char* CSGFoundry::RELDIR = "CSGFoundry"  ;


const char* CSGFoundry::getBaseDir(bool create) const
{
    int create_dirs = create ? 2 : 0 ; // 2:dirpath 0:noop
    const char* fold = geom ? SPath::Resolve(BASE, geom, create_dirs ) : nullptr ;
    const char* cfbase = SSys::getenvvar("CFBASE", fold  );  
    return cfbase ? strdup(cfbase) : nullptr ; 
}

void CSGFoundry::write() const 
{
    const char* cfbase = getBaseDir(true) ; 
    if( cfbase == nullptr )
    {
        LOG(fatal) << "cannot write unless CFBASE envvar defined or geom has been set " ; 
        return ;   
    }
    write(cfbase, RELDIR );  
}

void CSGFoundry::write(const char* base, const char* rel) const 
{
    std::stringstream ss ;   
    ss << base << "/" << rel ; 
    std::string dir = ss.str();   
    write(dir.c_str()); 
}

/**
CSGFoundry::write
------------------

Have observed that whilst changing geometry this can lead to "mixed" exports 
with the contents of CSGFoundry directory containing arrays from multiple exports. 
The inconsistency causes crashes.  

TODO: find way to avoid this, by deleting the folder ahead : or asserting on consistent time stamps
on loading 
 
**/
void CSGFoundry::write(const char* dir_) const 
{
    const char* dir = SPath::Resolve(dir_, DIRPATH); 
    LOG(info) << dir ; 

    if(meshname.size() > 0 ) NP::WriteNames( dir, "meshname.txt", meshname );
    if(mmlabel.size() > 0 )  NP::WriteNames( dir, "mmlabel.txt", mmlabel );
    if(hasMeta())  NP::WriteString( dir, "meta.txt", meta.c_str() ); 
              
    if(solid.size() > 0 ) NP::Write(dir, "solid.npy",  (int*)solid.data(),  solid.size(), 3, 4 ); 
    if(prim.size() > 0 ) NP::Write(dir, "prim.npy",   (float*)prim.data(), prim.size(),   4, 4 ); 
    if(node.size() > 0 ) NP::Write(dir, "node.npy",   (float*)node.data(), node.size(),   4, 4 ); 
    if(plan.size() > 0 ) NP::Write(dir, "plan.npy",   (float*)plan.data(), plan.size(),   1, 4 ); 
    if(tran.size() > 0 ) NP::Write(dir, "tran.npy",   (float*)tran.data(), tran.size(),   4, 4 ); 
    if(itra.size() > 0 ) NP::Write(dir, "itra.npy",   (float*)itra.data(), itra.size(),   4, 4 ); 
    if(inst.size() > 0 ) NP::Write(dir, "inst.npy",   (float*)inst.data(), inst.size(),   4, 4 ); 

    if(bnd)  bnd->save(dir,  "bnd.npy") ; 
    if(optical) optical->save(dir, "optical.npy") ; 
    if(icdf) icdf->save(dir, "icdf.npy") ; 
}

/**
CSGFoundry::saveOpticalBnd
----------------------------

CAUTION : ONLY APPROPRIATE IN SMALL SCALE TESTING WHEN ARE DOING 
DIRTY THINGS LIKE ADDING BOUNDARIES WITH QBnd::Add SEE FOR EXAMPLE
CSGOptiX/tests/CXRaindropTest.cc 

**/

void CSGFoundry::saveOpticalBnd() const 
{
    bool has_optical_bnd = bnd != nullptr && optical != nullptr ; 
    if(has_optical_bnd == false)
    {
        LOG(fatal) << "has_optical_bnd " << has_optical_bnd ; 
        return ;  
    }

    const char* ocf = getOriginCFBase(); 
    assert(ocf)  ; 
    const char* dir = SPath::Resolve(ocf, RELDIR, DIRPATH );
    LOG(info)
        << " save bnd.npy to dir [" << dir << "]" 
        << " originCFBase " << ocf  
        ;

    optical->save(dir,  "optical.npy" ); 
    bnd->save(dir,  "bnd.npy" ); 
}


template <typename T> void CSGFoundry::setMeta( const char* key, T value ){ NP::SetMeta(meta, key, value ); }

template void CSGFoundry::setMeta<int>(const char*, int );
template void CSGFoundry::setMeta<unsigned>(const char*, unsigned );
template void CSGFoundry::setMeta<float>(const char*, float );
template void CSGFoundry::setMeta<double>(const char*, double );
template void CSGFoundry::setMeta<std::string>(const char*, std::string );

template <typename T> T CSGFoundry::getMeta( const char* key, T fallback){ return NP::GetMeta(meta, key, fallback );  }

template int         CSGFoundry::getMeta<int>(const char*, int );
template unsigned    CSGFoundry::getMeta<unsigned>(const char*, unsigned );
template float       CSGFoundry::getMeta<float>(const char*, float );
template double      CSGFoundry::getMeta<double>(const char*, double );
template std::string CSGFoundry::getMeta<std::string>(const char*, std::string );

bool CSGFoundry::hasMeta() const {  return meta.empty() == false ; }


void CSGFoundry::load() 
{
    const char* cfbase = getBaseDir(false) ; 
    if( cfbase == nullptr )
    {
        LOG(fatal) << "cannot load unless CFBASE envvar defined or geom has been set " ; 
        return ;   
    }
    load(cfbase, RELDIR );  
}



void CSGFoundry::load(const char* base, const char* rel) 
{
    setCFBase(base);  

    std::stringstream ss ;   
    ss << base << "/" << rel ; 
    std::string dir = ss.str();   
    load(dir.c_str()); 
}

void CSGFoundry::setCFBase( const char* cfbase_ )
{
    cfbase = strdup(cfbase_); 
}
const char* CSGFoundry::getCFBase() const 
{
   return cfbase ; 
}
const char* CSGFoundry::getOriginCFBase() const 
{
   return origin ? origin->cfbase : cfbase ; 
}

/**
CSGFoundry::load
------------------

**/

void CSGFoundry::load( const char* dir_ )
{
    const char* dir = SPath::Resolve(dir_, NOOP ); 
    bool readable = SPath::IsReadable(dir); 
    if( readable == false )
    {
        LOG(fatal) << " dir is not readable " << dir ; 
        return ; 
    } 

    LOG(LEVEL) << dir ; 
    loaddir = strdup(dir) ; 

    NP::ReadNames( dir, "meshname.txt", meshname );  
    NP::ReadNames( dir, "mmlabel.txt", mmlabel );  

    const char* meta_str = NP::ReadString( dir, "meta.txt" ) ; 
    if(meta_str)
    {
       meta = meta_str ; 
    }
    else
    {
       LOG(warning) << " no meta.txt at " << dir ;  
    }

    loadArray( solid , dir, "solid.npy" ); 
    loadArray( prim  , dir, "prim.npy" ); 
    loadArray( node  , dir, "node.npy" ); 
    loadArray( tran  , dir, "tran.npy" ); 
    loadArray( itra  , dir, "itra.npy" ); 
    loadArray( inst  , dir, "inst.npy" ); 
    loadArray( plan  , dir, "plan.npy" , true );  
    // plan.npy loading optional, as only geometries with convexpolyhedrons such as trapezoids, tetrahedrons etc.. have them 

    if(NP::Exists(dir, "icdf.npy")) icdf = NP::Load(dir, "icdf.npy"); 

    if(NP::Exists(dir, "bnd.npy") && NP::Exists(dir, "optical.npy"))  
    {
        NP* optical_ = NP::Load(dir, "optical.npy"); 
        NP* bnd_     = NP::Load(dir, "bnd.npy"); 
        setOpticalBnd(optical_, bnd_);       // instanciates bd CSGName using bnd.names
    }

}

void CSGFoundry::setGeom(const char* geom_)
{
    geom = geom_ ? strdup(geom_) : nullptr ; 
}
void CSGFoundry::setOrigin(const CSGFoundry* origin_)
{
    origin = origin_ ; 
}
void CSGFoundry::setElv(const SBitSet* elv_)
{
    elv = elv_ ; 
}


/**
CSGFoundry::MakeGeom
----------------------

Intended for creation of small CSGFoundry test geometries that are entirely created by this method.

**/

CSGFoundry*  CSGFoundry::MakeGeom(const char* geom) // static
{
    CSGFoundry* fd = new CSGFoundry();  
    CSGMaker* mk = fd->maker ;
    CSGSolid* so = mk->make( geom );
    fd->setGeom(geom);  

    fd->addTranPlaceholder();  
    fd->addInstancePlaceholder(); 

    // avoid tripping some checks 
    fd->addMeshName(geom);   
    fd->addSolidLabel(geom);  
    fd->setMeta<std::string>("creator", SProc::ExecutableName() ); 
    fd->setMeta<std::string>("source", "CSGFoundry::MakeGeom" ); 

    assert( so ); 

    LOG(info) << " so " << so ;
    LOG(info) << " so.desc " << so->desc() ;
    LOG(info) << " fd.desc " << fd->desc() ;

    return fd ; 
}

CSGFoundry*  CSGFoundry::MakeDemo()
{
    CSGFoundry* fd = new CSGFoundry();
    fd->makeDemoSolids(); 
    return fd ; 
}




/**
CSGFoundry::Load
-------------------


**/
CSGFoundry* CSGFoundry::Load() // static
{
    CSGFoundry* src = CSGFoundry::Load_() ; 
    if(src == nullptr) return nullptr ; 
   
    const SBitSet* elv = SBitSet::Create( src->getNumMeshName(), "ELV", nullptr ); 

    if(elv)
    {
        LOG(info) << elv->desc() << std::endl << src->descELV(elv) ; 
    }

    CSGFoundry* dst = CSGCopy::Select(src, elv ); 

    dst->setOrigin(src); 
    dst->setElv(elv); 

    return dst ; 
}




/**
CSGFoundry::Load_
-------------------

The geometry loaded is sensitive to envvars GEOM, CFBASE, OPTICKS_KEY, OPTICKS_GEOCACHE_PREFIX 

Load precedence:

0. when GEOM envvar is defined the CSGFoundry directory beneath $TMP will be loaded 
1. when CFBASE envvar is defined the CSGFoundry directory within CFBASE dir will be loaded
2. otherwise SOpticksResource::CFBase will invoke CGDir to obtain the CSG_GGeo directory 
   corresponding to the current OPTICKS_KEY envvar 

**/

CSGFoundry* CSGFoundry::Load_() // static
{
    const char* cfbase = SOpticksResource::CFBase("CFBASE") ;  
    bool readable = SPath::IsReadable(cfbase, "CSGFoundry") ; 
    if(readable == false)
    {
        LOG(fatal) << " cfbase/CSGFoundy directory " << cfbase << "/CSGFoundry" << " IS NOT READABLE " ; 
        return nullptr ; 
    }

    CSGFoundry* fd = Load(cfbase, "CSGFoundry"); 
    return fd ; 
}

CSGFoundry*  CSGFoundry::LoadGeom(const char* geom) // static
{
    if(geom == nullptr) geom = SSys::getenvvar("GEOM", "GeneralSphereDEV") ; 
    CSGFoundry* fd = new CSGFoundry();  
    fd->setGeom(geom); 
    fd->load(); 

    if(VERBOSE > 0 )
    {
        if( !fd->meta.empty() )
        {
            LOG(info) << " geom " << geom << " loaddir " << fd->loaddir << std::endl << fd->meta ; 
        }
        else
        {
            LOG(info) << " geom " << geom << " loaddir " << fd->loaddir ;  
        }
    }
    return fd ; 
} 

CSGFoundry*  CSGFoundry::Load(const char* base, const char* rel) // static
{
    bool conventional = strcmp( rel, RELDIR) == 0  ; 
    if(!conventional) LOG(fatal) << "Convention is for rel to be named [" << RELDIR << "] not: [" << rel << "]"  ; 
    assert(conventional); 
    CSGFoundry* fd = new CSGFoundry();  
    fd->load(base, rel); 
    return fd ; 
} 

void CSGFoundry::setFold(const char* fold_)
{
    const char* rel = SPath::Basename(fold_); 
    assert( strcmp( rel, "CSGFoundry" ) == 0 ); 

    fold = strdup(fold_);  
    cfbase = SPath::Dirname(fold_) ; 
}

const char* CSGFoundry::getFold() const 
{
    return fold ; 
}




template<typename T>
void CSGFoundry::loadArray( std::vector<T>& vec, const char* dir, const char* name, bool optional )
{
    bool exists = NP::Exists(dir, name); 
    if(optional && !exists ) return ; 

    NP* a = NP::Load(dir, name);
    if(a == nullptr)   
    { 
        LOG(fatal) << "FAIL to load non-optional array  " << dir <<  "/" << name ; 
        LOG(fatal) << "convert geocache into CSGFoundry model using CSG_GGeo/run.sh " ; 
        // TODO: the CSGFoundry model should live inside the geocache rather than in tmp to avoid having to redo this frequently 
        assert(0); 
    }
    else
    { 
        assert( a->shape.size()  == 3 ); 
        unsigned ni = a->shape[0] ; 
        unsigned nj = a->shape[1] ; 
        unsigned nk = a->shape[2] ; 

        LOG(LEVEL) << " ni " << std::setw(5) << ni << " nj " << std::setw(1) << nj << " nk " << std::setw(1) << nk << " " << name ; 

        vec.clear(); 
        vec.resize(ni); 
        memcpy( vec.data(),  a->bytes(), sizeof(T)*ni ); 
    }
}

template void CSGFoundry::loadArray( std::vector<CSGSolid>& , const char* , const char*, bool ); 
template void CSGFoundry::loadArray( std::vector<CSGPrim>& , const char* , const char* , bool ); 
template void CSGFoundry::loadArray( std::vector<CSGNode>& , const char* , const char* , bool ); 
template void CSGFoundry::loadArray( std::vector<float4>& , const char* , const char*  , bool ); 
template void CSGFoundry::loadArray( std::vector<qat4>& , const char* , const char* , bool ); 


/**
CSGFoundry::upload
--------------------

Notice that the solid, inst and tran are not uploaded, as they are not needed on GPU. 
The reason is that the solid feeds into the GAS, the inst into the IAS and the tran 
are not needed because the inverse transforms are all that is needed.


**/

void CSGFoundry::upload()
{
    inst_find_unique(); 
    LOG(info) << "[ " << desc() ; 
    assert( tran.size() == itra.size() ); 

    // allocates and copies
    d_prim = prim.size() > 0 ? CU::UploadArray<CSGPrim>(prim.data(), prim.size() ) : nullptr ; 
    d_node = node.size() > 0 ? CU::UploadArray<CSGNode>(node.data(), node.size() ) : nullptr ; 
    d_plan = plan.size() > 0 ? CU::UploadArray<float4>(plan.data(), plan.size() ) : nullptr ; 
    d_itra = itra.size() > 0 ? CU::UploadArray<qat4>(itra.data(), itra.size() ) : nullptr ; 

    LOG(info) << "]"  ; 
}

void CSGFoundry::inst_find_unique()
{
    qat4::find_unique( inst, ins, gas, ias ); 
}

unsigned CSGFoundry::getNumUniqueIAS() const
{
    return ias.size(); 
}
unsigned CSGFoundry::getNumUniqueGAS() const
{
    return gas.size(); 
}
unsigned CSGFoundry::getNumUniqueINS() const
{
    return ins.size(); 
}



unsigned CSGFoundry::getNumInstancesIAS(unsigned ias_idx, unsigned long long emm) const
{
    return qat4::count_ias(inst, ias_idx, emm );  
}
void CSGFoundry::getInstanceTransformsIAS(std::vector<qat4>& select_inst, unsigned ias_idx, unsigned long long emm ) const 
{
    qat4::select_instances_ias(inst, select_inst, ias_idx, emm ) ;
}


unsigned CSGFoundry::getNumInstancesGAS(unsigned gas_idx) const
{
    return qat4::count_gas(inst, gas_idx );  
}

void CSGFoundry::getInstanceTransformsGAS(std::vector<qat4>& select_qv, unsigned gas_idx ) const 
{
    qat4::select_instances_gas(inst, select_qv, gas_idx ) ;
}

void CSGFoundry::getInstancePointersGAS(std::vector<const qat4*>& select_qi, unsigned gas_idx ) const 
{
    qat4::select_instance_pointers_gas(inst, select_qi, gas_idx ) ;
}

const qat4* CSGFoundry::getInstanceGAS(unsigned gas_idx_ , unsigned ordinal) const 
{
    int index = qat4::find_instance_gas(inst, gas_idx_, ordinal);
    return index > -1 ? &inst[index] : nullptr ; 
}

std::string CSGFoundry::descGAS() const 
{
    std::stringstream ss ; 
    ss << desc() << std::endl ; 
    for(unsigned i=0 ; i < gas.size() ; i++)
    {   
        unsigned gas_idx = gas[i]; 
        unsigned num_inst_gas = getNumInstancesGAS(gas_idx); 
        ss << std::setw(5) << gas_idx << ":" << std::setw(8) << num_inst_gas << std::endl ;  
    }   
    std::string s = ss.str(); 
    return s ; 
}



/**
CSGFoundry::parseMOI
-------------------------

MOI lookups Meshidx-meshOrdinal-Instanceidx, examples of moi strings::

   sWorld:0:0
   sWorld:0
   sWorld

   0:0:0
   0:0
   0

**/
void CSGFoundry::parseMOI(int& midx, int& mord, int& iidx, const char* moi) const 
{
    id->parseMOI(midx, mord, iidx, moi ); 
}
const char* CSGFoundry::getName(unsigned midx) const 
{
    return id->getName(midx); 
}

/**
CSGFoundry::getCenterExtent
-------------------------------

For midx -1 returns ce obtained from the ias bbox, 
otherwise uses CSGTarget to lookup the center extent. 

For global geometry which typically means a default iidx of 0 
there is special handling for iidx -1/-2/-3 in CSGTarget::getCenterExtent

iidx -1
    uses getLocalCenterExtent

iidx -2
    uses SCenterExtentFrame xyzw : ordinary XYZ frame 

iidx -3
    uses SCenterExtentFrame rtpw : tangential RTP frame 


**/

int CSGFoundry::getCenterExtent(float4& ce, int midx, int mord, int iidx, qat4* m2w, qat4* w2m  ) const 
{
    int rc = 0 ; 
    if( midx == -1 )
    { 
        unsigned long long emm = 0ull ;   // hmm instance var ?
        iasCE(ce, emm); 
    }
    else
    {
        rc = target->getCenterExtent(ce, midx, mord, iidx, m2w, w2m );    // should use emm ?
    }

    if( rc != 0 )
    {
        LOG(error) << " non-zero RC from CSGTarget::getCenterExtent " ;   
    }
    return rc ; 
}


int CSGFoundry::getTransform(qat4& q, int midx, int mord, int iidx) const 
{
    return target->getTransform(q, midx, mord, iidx); 
}





/**
CSGFoundry::kludgeScalePrimBBox
---------------------------------

**/

void CSGFoundry::kludgeScalePrimBBox( const char* label, float dscale )
{
    std::vector<unsigned> solidIdx ; 
    findSolidIdx(solidIdx, label); 

    for(int i=0 ; i < int(solidIdx.size()) ; i++)
    {
        unsigned soIdx = solidIdx[i]; 
        kludgeScalePrimBBox( soIdx, dscale ); 
    }
}

/**
CSGFoundry::kludgeScalePrimBBox
--------------------------------

Scaling the AABB of all *CSGPrim* of the *solidIdx* 

**/

void CSGFoundry::kludgeScalePrimBBox( unsigned solidIdx, float dscale )
{
    CSGSolid* so = solid.data() + solidIdx ; 
    so->type = KLUDGE_BBOX_SOLID ; 

    unsigned primOffset = so->primOffset ;
    unsigned numPrim = so->numPrim ; 

    for(unsigned primIdx=0 ; primIdx < numPrim ; primIdx++)
    {
        // primIdx                :   0,1,2,3,...,numPrim-1 
        // numPrim-1 - primIdx    :  numPrim-1, numPrim-1-1, numPrim-1-2, ... , 0          
        // scale                  :  1+(numPrim-1)*dscale, 

        float scale = 1.f + dscale*float(numPrim - 1u - primIdx) ;  
        LOG(info) << " primIdx " << std::setw(2) << primIdx << " scale " << scale ; 
        std::cout 
            << "CSGFoundry::kludgeScalePrimBBox" 
            << " primIdx " << std::setw(2) << primIdx 
            << " numPrim " << std::setw(2) << numPrim
            << " scale " << scale 
            << std::endl 
            ; 
        CSGPrim* pr = prim.data() + primOffset + primIdx ;  // about to modify, so low level access 
        pr->scaleAABB_(scale); 
    } 
}


/*
void CSGFoundry::dumpPrimBoundary( const CSGPrim* prim ) const 
{

}
*/


