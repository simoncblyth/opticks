#include <iostream>
#include <iomanip>
#include <array>
#include <vector>
#include <algorithm>
#include <cstring>

#include <glm/glm.hpp>
#include <glm/gtx/string_cast.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "SStr.hh"
#include "SPath.hh"
#include "NP.hh"
#include "PLOG.hh"

#include "sutil_vec_math.h"
#include "OpticksCSG.h"
#include "CSGSolid.h"
#include "CU.h"
#include "CSGFoundry.h"
#include "CSGName.h"
#include "CSGTarget.h"
#include "AABB.h"

const unsigned CSGFoundry::IMAX = 50000 ; 

const plog::Severity CSGFoundry::LEVEL = PLOG::EnvLevel("CSGFoundry", "DEBUG" ); 

CSGFoundry::CSGFoundry()
    :
    d_prim(nullptr),
    d_node(nullptr),
    d_plan(nullptr),
    d_itra(nullptr),
    id(new CSGName(this)),
    target(new CSGTarget(this)),
    deepcopy_everynode_transform(true),
    last_added_solid(nullptr),
    last_added_prim(nullptr),
    bnd(nullptr),
    icdf(nullptr)
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

void CSGFoundry::makeDemoSolids()
{
    makeSphere(); 
    makeZSphere(); 
    makeCone(); 
    makeHyperboloid(); 
    makeBox3(); 
    makePlane(); 
    makeSlab(); 
    makeCylinder() ; 
    makeDisc(); 
    makeConvexPolyhedronCube(); 
    makeConvexPolyhedronTetrahedron(); 
    makeEllipsoid(); 
    makeUnionBoxSphere();
    makeIntersectionBoxSphere();
    makeDifferenceBoxSphere();
    makeRotatedCylinder();
}

void CSGFoundry::makeDemoGrid()
{
    makeDemoSolids(); 
    unsigned num_solids = getNumSolid(); 
    LOG(info) << " num_solids " << num_solids ; 

    float gridscale = 100.f ; 
    std::array<int,9> grid = { -10,11,2,  -10,11,2, -10,11,2  } ;

    unsigned ias_idx = 0 ; 
    unsigned count = 0 ; 

    for(int i=grid[0] ; i < grid[1] ; i+=grid[2] ){
    for(int j=grid[3] ; j < grid[4] ; j+=grid[5] ){
    for(int k=grid[6] ; k < grid[7] ; k+=grid[8] ){

        qat4 instance  ;   
        instance.q3.f.x = float(i)*gridscale ; 
        instance.q3.f.y = float(j)*gridscale ; 
        instance.q3.f.z = float(k)*gridscale ; 
        
        unsigned ins_idx = inst.size() ;    
        unsigned gas_idx = count % num_solids ; 

        instance.setIdentity( ins_idx, gas_idx, ias_idx );  
        inst.push_back( instance );  
     
        count++ ; 
    }   
    }   
    }   
}

std::string CSGFoundry::desc() const 
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
       << " num_prim " << prim.size()
       << " num_node " << node.size()
       << " num_plan " << plan.size()
       << " num_tran " << tran.size()
       << " num_itra " << itra.size()
       << " num_inst " << inst.size()
       << " ins " << ins.size()
       << " gas " << gas.size()
       << " ias " << ias.size()
       ;

    return ss.str(); 
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
    assert( mismatch == 0 ); 
    return mismatch ; 
}

template<typename T>
int CSGFoundry::CompareVec( const char* name, const std::vector<T>& a, const std::vector<T>& b )
{
    int mismatch = 0 ; 

    bool size_match = a.size() == b.size() ; 
    if(!size_match) LOG(info) << name << " size_match FAIL " ; 
    if(!size_match) mismatch += 1 ; 

    int data_match = memcmp( a.data(), b.data(), a.size()*sizeof(T) ) ; 
    if(data_match != 0) LOG(info) << name << " sizeof(T) " << sizeof(T) << " data_match FAIL " ; 
    if(data_match != 0) mismatch += 1 ; 

    int byte_match = CompareBytes( a.data(), b.data(), a.size()*sizeof(T) ) ;
    if(byte_match != 0) LOG(info) << name << " sizeof(T) " << sizeof(T) << " byte_match FAIL " ; 
    if(byte_match != 0) mismatch += 1 ; 

    if( mismatch != 0 ) LOG(fatal) << " mismatch FAIL for " << name ;  
    if( mismatch != 0 ) std::cout << " mismatch FAIL for " << name << std::endl ;  
    assert( mismatch == 0 ); 
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


// bbox of the IAS by tranforming the center_extent cubes of all instances
// hmm: could get a smaller bbox by using the bbox and not the ce of the instances 
// need to add bb to solid...
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

void CSGFoundry::dumpSolid(unsigned solidIdx) const 
{
    const CSGSolid* so = solid.data() + solidIdx ; 
    LOG(info) << so->desc() ; 

    for(unsigned primIdx=so->primOffset ; primIdx < so->primOffset+so->numPrim ; primIdx++)
    {
        const CSGPrim* pr = prim.data() + primIdx ; 
        LOG(info) 
            << " primIdx " << std::setw(3) << primIdx << " "
            << pr->desc() 
            ; 

        for(unsigned nodeIdx=pr->nodeOffset() ; nodeIdx < pr->nodeOffset()+pr->numNode() ; nodeIdx++)
        {
            const CSGNode* nd = node.data() + nodeIdx ; 
            LOG(info) << nd->desc() ; 
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

    for(unsigned primIdx=so->primOffset ; primIdx < so->primOffset+so->numPrim ; primIdx++)  
    {
        const CSGPrim* pr = getPrim(primIdx) ;  // note absolute primIdx
        assert(pr) ; 
        ss << "    primIdx " << std::setw(5) << primIdx << " : " << pr->desc() << std::endl ; 
    } 

    std::string s = ss.str(); 
    return s ; 
}


void CSGFoundry::dumpPrim(unsigned solidIdx) const 
{
    std::string s = descPrim(solidIdx); 
    LOG(info) << std::endl << s ;
}

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
    const CSGPrim* pr0 = prim.data() + so->primOffset ; 
    const CSGNode* nd0 = node.data() + pr0->nodeOffset() ;  

    std::stringstream ss ;
    ss << std::endl << so->desc() << std::endl  ;

    for(unsigned primIdx=so->primOffset ; primIdx < so->primOffset+so->numPrim ; primIdx++)
    {
        const CSGPrim* pr = prim.data() + primIdx ; 
        int numNode = pr->numNode() ; 
        for(unsigned nodeIdx=pr->nodeOffset() ; nodeIdx < pr->nodeOffset()+numNode ; nodeIdx++)
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
    const CSGPrim* pr0 = prim.data() + so->primOffset ; 
    const CSGNode* nd0 = node.data() + pr0->nodeOffset() ;  

    std::stringstream ss ;
    ss << std::endl << so->desc() << std::endl  ;

    for(unsigned primIdx=so->primOffset ; primIdx < so->primOffset+so->numPrim ; primIdx++)
    {
        const CSGPrim* pr = prim.data() + primIdx ; 
        int numNode = pr->numNode() ; 
        for(unsigned nodeIdx=pr->nodeOffset() ; nodeIdx < pr->nodeOffset()+numNode ; nodeIdx++)
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




CSGSolid* CSGFoundry::make(const char* name)
{
    CSGSolid* so = nullptr ; 
    if(     strcmp(name, "sphe") == 0) so = makeSphere(name) ;
    else if(strcmp(name, "zsph") == 0) so = makeZSphere(name) ;
    else if(strcmp(name, "cone") == 0) so = makeCone(name) ;
    else if(strcmp(name, "hype") == 0) so = makeHyperboloid(name) ;
    else if(strcmp(name, "box3") == 0) so = makeBox3(name) ;
    else if(strcmp(name, "plan") == 0) so = makePlane(name) ;
    else if(strcmp(name, "slab") == 0) so = makeSlab(name) ;
    else if(strcmp(name, "cyli") == 0) so = makeCylinder(name) ;
    else if(strcmp(name, "disc") == 0) so = makeDisc(name) ;
    else if(strcmp(name, "vcub") == 0) so = makeConvexPolyhedronCube(name) ;
    else if(strcmp(name, "vtet") == 0) so = makeConvexPolyhedronTetrahedron(name) ;
    else if(strcmp(name, "elli") == 0) so = makeEllipsoid(name) ;
    else if(strcmp(name, "ubsp") == 0) so = makeUnionBoxSphere(name) ;
    else if(strcmp(name, "ibsp") == 0) so = makeIntersectionBoxSphere(name) ;
    else if(strcmp(name, "dbsp") == 0) so = makeDifferenceBoxSphere(name) ;
    else if(strcmp(name, "rcyl") == 0) so = makeRotatedCylinder(name) ;
    else LOG(fatal) << "invalid name [" << name << "]" ; 
    assert( so ); 
    return so ;  
}


/**
CSGFoundry::addNode
--------------------

**/

CSGNode* CSGFoundry::addNode(CSGNode nd, const std::vector<float4>* pl )
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

    node.push_back(nd); 
    last_added_node = node.data() + globalNodeIdx ;
    return last_added_node ; 
}

float4* CSGFoundry::addPlan(const float4& pl )
{
    unsigned idx = plan.size(); 
    assert( idx < IMAX ); 
    plan.push_back(pl); 
    return plan.data() + idx ; 
}

template<typename T>
unsigned CSGFoundry::addTran( const Tran<T>& tr  )
{
    //LOG(info) << tr.brief() ; 
    qat4 t(glm::value_ptr(tr.t));  // narrowing when T=double
    qat4 v(glm::value_ptr(tr.v)); 
    unsigned idx = addTran(&t, &v); 
    return idx ; 
}

template unsigned CSGFoundry::addTran<float>(const Tran<float>& ) ;
template unsigned CSGFoundry::addTran<double>(const Tran<double>& ) ;


unsigned CSGFoundry::addTran( const qat4* tr, const qat4* it )
{
    unsigned idx = tran.size();   // size before push_back 
    assert( tran.size() == itra.size()) ; 
    tran.push_back(*tr); 
    itra.push_back(*it); 
    return idx ;  
}

CSGNode* CSGFoundry::addNodes(const std::vector<CSGNode>& nds )
{
    unsigned idx = node.size() ; 
    for(unsigned i=0 ; i < nds.size() ; i++) 
    {
        const CSGNode& nd = nds[i]; 
        idx = node.size() ;  
        assert( idx < IMAX ); 
        node.push_back(nd); 
    }
    return node.data() + idx ; 
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
    //pr.setSbtIndexOffset(globalPrimIdx) ;  // <--- bug ?
    pr.setSbtIndexOffset(localPrimIdx) ; 

    pr.setMeshIdx(-1) ;                // metadata, that may be set by caller 

    pr.setTranOffset(tran.size());     // HMM are tranOffset and planOffset used now that use global referencing  ?
    pr.setPlanOffset(plan.size()); 

    assert( globalPrimIdx < IMAX ); 
    prim.push_back(pr); 

    last_added_prim = prim.data() + globalPrimIdx ;
    last_added_node = nullptr ; 

    return last_added_prim  ; 
}


// collect Prims with the supplied mesh_idx 
void CSGFoundry::getMeshPrim(std::vector<CSGPrim>& select_prim, unsigned mesh_idx ) const 
{
    CSGPrim::select_prim_mesh(prim, select_prim, mesh_idx); 
}
unsigned CSGFoundry::getNumMeshPrim(unsigned mesh_idx ) const 
{
    return CSGPrim::count_prim_mesh(prim, mesh_idx); 
}

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

    AABB sbb = {} ; 
    CSGSolid* cso = addSolid(numPrim, cso_label.c_str()); 
    cso->type = DEEP_COPY_SOLID ; 

    for(unsigned primIdx=oso->primOffset ; primIdx < oso->primOffset+oso->numPrim ; primIdx++)
    {
        const CSGPrim* opr = prim.data() + primIdx ; 

        unsigned numNode = opr->numNode()  ; 
        int nodeOffset_ = -1 ; // as deep copying, -1 declares that will immediately add numNode new nodes

        AABB pbb = {} ; 
        CSGPrim* cpr = addPrim(numNode, nodeOffset_ );

        cpr->setMeshIdx(opr->meshIdx());    // copy the metadata that makes sense to be copied 
        cpr->setRepeatIdx(opr->repeatIdx()); 
        cpr->setPrimIdx(opr->primIdx()); 

        for(unsigned nodeIdx=opr->nodeOffset() ; nodeIdx < opr->nodeOffset()+opr->numNode() ; nodeIdx++)
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
            bool c0 = cnd.complement(); 
            cnd.zeroTransformComplement(); 
            cnd.setComplement(c0) ; 
            cnd.setTransform( c_tranIdx );   

            unsigned c_tranIdx2 = cnd.gtransformIdx() ; 

            bool match = c_tranIdx2 == c_tranIdx ;  
            if(!match) std::cout << "set/get transform fail c_tranIdx " << c_tranIdx << " c_tranIdx2 " << c_tranIdx2 << std::endl ; 
            assert(match); 


            cnd.setAABBLocal() ;  // reset to local with no transform applied
            if(tra)  
            {
                tra->transform_aabb_inplace( cnd.AABB() ); 
            }
            pbb.include_aabb( cnd.AABB() ); 
            addNode(cnd);       
        }                  // over nodes of the Prim 
         
        cpr->setAABB(pbb.data()); 
        sbb.include_aabb(pbb.data()) ; 
    }    // over Prim of the Solid

    cso->center_extent = sbb.center_extent() ;  
    return cso ; 
}


/**
Foundary::makeLayered
----------------------------

Once have transforms working can generalize to any shape. 
But prior to that just do layering for sphere for adiabatic transition
from Shape to CSGFoundry/CSGSolid.

NB Each layer is a separate CSGPrim with a single CSGNode 

NB the ordering of addition is prescribed, must stick 
ridgidly to the below order of addition.  

   addSolid
   addPrim
   addNode

Note that Node and Prim can be created anytime, the 
restriction is on the order of addition because 
of the capturing of offsets.

**/

CSGSolid* CSGFoundry::makeLayered(const char* label, float outer_radius, unsigned layers )
{
    std::vector<float> radii ;
    for(unsigned i=0 ; i < layers ; i++) radii.push_back(outer_radius*float(layers-i)/float(layers)) ; 

    unsigned numPrim = layers ; 
    CSGSolid* so = addSolid(numPrim, label); 
    so->center_extent = make_float4( 0.f, 0.f, 0.f, outer_radius ) ; 

    for(unsigned i=0 ; i < numPrim ; i++)
    {
        unsigned numNode = 1 ; 
        int nodeOffset_ = -1 ; 
        CSGPrim* pr = addPrim(numNode, nodeOffset_ );
 
        float radius = radii[i]; 

        CSGNode* nd = nullptr ; 

        if(strcmp(label, "sphere") == 0)
        {
            nd = addNode(CSGNode::Sphere(radius)); 
        }
        else if(strcmp(label, "zsphere") == 0)
        {
            nd = addNode(CSGNode::ZSphere(radius, -radius/2.f , radius/2.f )); 
        }
        else
        {
            assert( 0 && "layered only implemented for sphere and zsphere currently" ); 
        } 

        // pr->setSbtIndexOffset(i) ;  // NOW done in addPrim
        pr->setAABB( nd->AABB() ); 
    }
    return so ; 
}


CSGSolid* CSGFoundry::makeScaled(const char* label, const char* demo_node_type, float outer_scale, unsigned layers )
{
    std::vector<float> scales ;
    for(unsigned i=0 ; i < layers ; i++) scales.push_back(outer_scale*float(layers-i)/float(layers)) ; 

    unsigned numPrim = layers ; 
    CSGSolid* so = addSolid(numPrim, label); 
    AABB bb = {} ; 

    for(unsigned i=0 ; i < numPrim ; i++)
    {
        unsigned numNode = 1 ; 
        int nodeOffset_ = -1; 
        CSGPrim* pr = addPrim(numNode, nodeOffset_); 
        CSGNode* nd = addNode(CSGNode::MakeDemo(demo_node_type)) ;
    
        float scale = scales[i]; 

        const Tran<double>* tran_scale = Tran<double>::make_scale( double(scale), double(scale), double(scale) ); 
        unsigned transform_idx = 1 + addTran(*tran_scale);      // 1-based idx, 0 meaning None

        nd->setTransform(transform_idx); 

        const qat4* tr = getTran(transform_idx-1u) ; 

        tr->transform_aabb_inplace( nd->AABB() ); 

        bb.include_aabb( nd->AABB() ); 

        // pr->setSbtIndexOffset(i) ;  //  NOW done in addPrim
        pr->setAABB( nd->AABB() ); 
    }

    so->center_extent = bb.center_extent() ;  
    LOG(info) << " so->center_extent " << so->center_extent ; 
    return so ; 
}





CSGSolid* CSGFoundry::makeClustered(const char* label,  int i0, int i1, int is, int j0, int j1, int js, int k0, int k1, int ks, double unit, bool inbox ) 
{
    unsigned numPrim = inbox ? 1 : 0 ; 
    for(int i=i0 ; i < i1 ; i+=is ) 
    for(int j=j0 ; j < j1 ; j+=js ) 
    for(int k=k0 ; k < k1 ; k+=ks ) 
    {
        //LOG(info) << std::setw(2) << numPrim << " (i,j,k) " << "(" << i << "," << j << "," << k << ") " ; 
        numPrim += 1 ; 
    }
       
    LOG(info) 
        << " label " << label  
        << " numPrim " << numPrim 
        << " inbox " << inbox
        ;  

    CSGSolid* so = addSolid(numPrim, label);
    unsigned idx = 0 ; 

    AABB bb = {} ; 
 
    for(int i=i0 ; i < i1 ; i+=is ) 
    for(int j=j0 ; j < j1 ; j+=js ) 
    for(int k=k0 ; k < k1 ; k+=ks ) 
    {
        unsigned numNode = 1 ; 
        int nodeOffset_ = -1 ;  // -1:use current node count as about to add the declared numNode
        CSGPrim* p = addPrim(numNode, nodeOffset_); 
        CSGNode* n = addNode(CSGNode::MakeDemo(label)) ;
    
        const Tran<double>* translate = Tran<double>::make_translate( double(i)*unit, double(j)*unit, double(k)*unit ); 
        unsigned transform_idx = 1 + addTran(*translate);      // 1-based idx, 0 meaning None
        n->setTransform(transform_idx); 
        const qat4* t = getTran(transform_idx-1u) ; 

        t->transform_aabb_inplace( n->AABB() ); 

        bb.include_aabb( n->AABB() ); 

        // p->setSbtIndexOffset(idx) ;    //  now done in addPrim
        p->setAABB( n->AABB() );  // HUH : THIS SHOULD BE bb ?

        //DumpAABB("p->AABB() aft setup", p->AABB() ); 
        
        LOG(info) << " idx " << idx << " transform_idx " << transform_idx ; 
 
        idx += 1 ; 
    }


    if(inbox)
    {
        float4 ce = bb.center_extent(); 
        float fullside = ce.w*2.f ; 

        const Tran<float>* to_center = Tran<float>::make_translate( float(ce.x), float(ce.y), float(ce.z) ); 
        unsigned transform_idx = 1 + addTran(*to_center);  // 1-based idx, 0 meaning None
        const qat4* t = getTran(transform_idx-1u) ; 

        unsigned numNode = 1 ; 
        int nodeOffset_ = -1 ;  // -1:use current node count as about to add the declared numNode
        CSGPrim* p = addPrim(numNode, nodeOffset_ ); 
        CSGNode bx = CSGNode::Box3(fullside) ;
        CSGNode* n = addNode(CSGNode::Box3(fullside)); 
        n->setTransform(transform_idx); 
        t->transform_aabb_inplace( n->AABB() ); 

        // p->setSbtIndexOffset(idx);   now done in addPrim
        p->setAABB( n->AABB() );
        
        idx += 1 ; 
    }


    so->center_extent = bb.center_extent() ;   // contains AABB of all CSGPrim 
    LOG(info) << " so->center_extent " << so->center_extent ; 
    return so ; 
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



/**
CSGFoundry::makeSolid11 makes 1-CSGPrim with 1-CSGNode
---------------------------------------------------------
**/

CSGSolid* CSGFoundry::makeSolid11(const char* label, CSGNode nd, const std::vector<float4>* pl, int meshIdx  ) 
{
    unsigned numPrim = 1 ; 
    CSGSolid* so = addSolid(numPrim, label);

    unsigned numNode = 1 ; 
    int nodeOffset_ = -1 ;  
    CSGPrim* p = addPrim(numNode, nodeOffset_); 
    p->setMeshIdx(meshIdx); 

    CSGNode* n = addNode(nd, pl ); 
    p->setAABB( n->AABB() ); 

    float extent = p->extent(); 
    if(extent == 0.f )
        LOG(fatal) << "FATAL : " << label << " : got zero extent " ; 
    assert( extent > 0.f ); 

    AABB bb = AABB::Make( p->AABB() ); 
    so->center_extent = bb.center_extent()  ; 
    LOG(info) << "so.label " << so->label << " so.center_extent " << so->center_extent ; 
    return so ; 
}

CSGSolid* CSGFoundry::makeBooleanBoxSphere( const char* label, char op_, float radius, float fullside, int meshIdx )
{
    CSGNode op = CSGNode::BooleanOperator(op_); 
    CSGNode bx = CSGNode::Box3(fullside) ; 
    CSGNode sp = CSGNode::Sphere(radius); 

    unsigned numPrim = 1 ; 
    CSGSolid* so = addSolid(numPrim, label);

    unsigned numNode = 3 ; 
    int nodeOffset_ = -1 ; 
    CSGPrim* p = addPrim(numNode, nodeOffset_ ); 
    p->setMeshIdx(meshIdx); 

    addNode(op); 
    addNode(bx); 
    addNode(sp); 
     
    AABB bb = {} ;
    bb.include_aabb( bx.AABB() );     // assumes any transforms have been applied to the Node AABB
    bb.include_aabb( sp.AABB() ); 
    p->setAABB( bb.data() );  

    so->center_extent = bb.center_extent()  ; 
    LOG(info) << "so.label " << so->label << " so.center_extent " << so->center_extent ; 
    return so ; 
}





CSGSolid* CSGFoundry::makeUnionBoxSphere( const char* label, float radius, float fullside ){
    return makeBooleanBoxSphere(label, 'U', radius, fullside, UBSP_MIDX ); 
}
CSGSolid* CSGFoundry::makeIntersectionBoxSphere( const char* label, float radius, float fullside ){
    return makeBooleanBoxSphere(label, 'I', radius, fullside, IBSP_MIDX ); 
}
CSGSolid* CSGFoundry::makeDifferenceBoxSphere( const char* label, float radius, float fullside ){
    return makeBooleanBoxSphere(label, 'D', radius, fullside, DBSP_MIDX ); 
}


CSGSolid* CSGFoundry::makeSphere(const char* label, float radius)
{
    CSGNode nd = CSGNode::Sphere(radius); 
    return makeSolid11(label, nd, nullptr, SPHE_MIDX ); 
}
CSGSolid* CSGFoundry::makeEllipsoid(  const char* label, float rx, float ry, float rz )
{
    CSGNode nd = CSGNode::Sphere(rx);

    double dx = double(rx) ; 
    double dy = double(ry) ; 
    double dz = double(rz) ; 
 
    double sx = double(1.) ; 
    double sy = dy/dx ; 
    double sz = dz/dx ; 

    const Tran<double>* tr = Tran<double>::make_scale(sx, sy, sz ); 

    unsigned idx = 1 + addTran(*tr);      // 1-based idx, 0 meaning None
    //LOG(info) << "CSGFoundry::makeEllipsoid " << *tr ;

    nd.setTransform(idx); 
    return makeSolid11(label, nd, nullptr, ELLI_MIDX ); 
}




CSGSolid* CSGFoundry::makeRotatedCylinder(const char* label, float px, float py, float radius, float z1, float z2, float ax, float ay, float az, float angle_deg )
{
    CSGNode nd = CSGNode::Cylinder( px, py, radius, z1, z2 ); 
    const Tran<float>* tr = Tran<float>::make_rotate(ax, ay, az, angle_deg ); 
    unsigned idx = 1 + addTran(*tr);      // 1-based idx, 0 meaning None
    //LOG(info) << *tr ;
    nd.setTransform(idx); 
    return makeSolid11(label, nd, nullptr, RCYL_MIDX ); 
}







CSGSolid* CSGFoundry::makeZSphere(const char* label, float radius, float z1, float z2)
{
    CSGNode nd = CSGNode::ZSphere(radius, z1, z2); 
    return makeSolid11(label, nd, nullptr, ZSPH_MIDX ); 
}




CSGSolid* CSGFoundry::makeCone(const char* label, float r1, float z1, float r2, float z2)
{
    CSGNode nd = CSGNode::Cone(r1, z1, r2, z2 ); 
    return makeSolid11(label, nd, nullptr, CONE_MIDX ); 
}

CSGSolid* CSGFoundry::makeHyperboloid(const char* label, float r0, float zf, float z1, float z2)
{
    CSGNode nd = CSGNode::Hyperboloid( r0, zf, z1, z2 ); 
    return makeSolid11(label, nd, nullptr, HYPE_MIDX ); 
}

CSGSolid* CSGFoundry::makeBox3(const char* label, float fx, float fy, float fz )
{
    CSGNode nd = CSGNode::Box3(fx, fy, fz); 
    return makeSolid11(label, nd, nullptr, BOX3_MIDX ); 
}

CSGSolid* CSGFoundry::makePlane(const char* label, float nx, float ny, float nz, float d)
{
    CSGNode nd = CSGNode::Plane(nx, ny, nz, d ); 
    return makeSolid11(label, nd, nullptr, PLAN_MIDX ); 
}

CSGSolid* CSGFoundry::makeSlab(const char* label, float nx, float ny, float nz, float d1, float d2 )
{
    CSGNode nd = CSGNode::Slab( nx, ny, nz, d1, d1 ); 
    return makeSolid11(label, nd, nullptr, SLAB_MIDX ); 
}

CSGSolid* CSGFoundry::makeCylinder(const char* label, float px, float py, float radius, float z1, float z2)
{
    CSGNode nd = CSGNode::Cylinder( px, py, radius, z1, z2 ); 
    return makeSolid11(label, nd, nullptr, CYLI_MIDX ); 
}

CSGSolid* CSGFoundry::makeDisc(const char* label, float px, float py, float ir, float r, float z1, float z2)
{
    CSGNode nd = CSGNode::Disc(px, py, ir, r, z1, z2 ); 
    return makeSolid11(label, nd, nullptr, DISC_MIDX ); 
}


float4 CSGFoundry::TriPlane( const std::vector<float3>& v, unsigned i, unsigned j, unsigned k )  // static 
{
    // normal for plane through v[i] v[j] v[k]
    float3 ij = v[j] - v[i] ; 
    float3 ik = v[k] - v[i] ; 
    float3 n = normalize(cross(ij, ik )) ;
    float di = dot( n, v[i] ) ;
    float dj = dot( n, v[j] ) ;
    float dk = dot( n, v[k] ) ;
    //LOG(info) << " di " << di << " dj " << dj << " dk " << dk << " n (" << n.x << "," << n.y << "," << n.z << ")" ; 
    float4 plane = make_float4( n, di ) ; 
    return plane ;  
}

CSGSolid* CSGFoundry::makeConvexPolyhedronCube(const char* label, float extent)
{
    float hx = extent ; 
    float hy = extent/2.f ; 
    float hz = extent/3.f ; 

    std::vector<float4> pl ; 
    pl.push_back( make_float4(  1.f,  0.f,  0.f, hx ) ); 
    pl.push_back( make_float4( -1.f,  0.f,  0.f, hx ) ); 
    pl.push_back( make_float4(  0.f,  1.f,  0.f, hy ) ); 
    pl.push_back( make_float4(  0.f, -1.f,  0.f, hy ) ); 
    pl.push_back( make_float4(  0.f,  0.f,  1.f, hz ) ); 
    pl.push_back( make_float4(  0.f,  0.f, -1.f, hz ) );

    CSGNode nd = {} ;
    nd.setAABB(-hx, -hy, -hz, hx, hy, hz); 
    return makeSolid11(label, nd, &pl, VCUB_MIDX ); 
}


/*  
     https://en.wikipedia.org/wiki/Tetrahedron

       0:(1,1,1)
       1:(1,−1,−1)
       2:(−1,1,−1) 
       3:(−1,−1,1)

                              (1,1,1)
                 +-----------0
                /|          /| 
     (-1,-1,1) / |         / |
              3-----------+  |
              |  |        |  |
              |  |        |  |
   (-1,1,-1)..|..2--------|--+
              | /         | /
              |/          |/
              +-----------1
                          (1,-1,-1)      

          Faces (right-hand-rule oriented outwards normals)
                0-1-2
                1-3-2
                3-0-2
                0-3-1

         z  y
         | /
         |/
         +---> x
*/

CSGSolid* CSGFoundry::makeConvexPolyhedronTetrahedron(const char* label, float extent)
{
    //extent = 100.f*sqrt(3); 
    float s = extent ; 

    std::vector<float3> vtx ; 
    vtx.push_back(make_float3( s, s, s));  
    vtx.push_back(make_float3( s,-s,-s)); 
    vtx.push_back(make_float3(-s, s,-s)); 
    vtx.push_back(make_float3(-s,-s, s)); 

    std::vector<float4> pl ; 
    pl.push_back(TriPlane(vtx, 0, 1, 2)) ;  
    pl.push_back(TriPlane(vtx, 1, 3, 2)) ;  
    pl.push_back(TriPlane(vtx, 3, 0, 2)) ;  
    pl.push_back(TriPlane(vtx, 0, 3, 1)) ;  

    //for(unsigned i=0 ; i < pl.size() ; i++) LOG(info) << " pl (" << pl[i].x << "," << pl[i].y << "," << pl[i].z << "," << pl[i].w << ") " ;
 
    CSGNode nd = {} ;
    nd.setAABB(extent); 
    return makeSolid11(label, nd, &pl, VTET_MIDX ); 
}

void CSGFoundry::write(const char* base, const char* rel) const 
{
    std::stringstream ss ;   
    ss << base << "/" << rel ; 
    std::string dir = ss.str();   
    write(dir.c_str()); 
}
void CSGFoundry::write(const char* dir) const 
{
    LOG(info) << dir ; 

    NP::WriteNames( dir, "name.txt", name );
              
    if(solid.size() > 0 ) NP::Write(dir, "solid.npy",  (int*)solid.data(),  solid.size(), 3, 4 ); 
    if(prim.size() > 0 ) NP::Write(dir, "prim.npy",   (float*)prim.data(), prim.size(),   4, 4 ); 
    if(node.size() > 0 ) NP::Write(dir, "node.npy",   (float*)node.data(), node.size(),   4, 4 ); 
    if(plan.size() > 0 ) NP::Write(dir, "plan.npy",   (float*)plan.data(), plan.size(),   1, 4 ); 
    if(tran.size() > 0 ) NP::Write(dir, "tran.npy",   (float*)tran.data(), tran.size(),   4, 4 ); 
    if(itra.size() > 0 ) NP::Write(dir, "itra.npy",   (float*)itra.data(), itra.size(),   4, 4 ); 
    if(inst.size() > 0 ) NP::Write(dir, "inst.npy",   (float*)inst.data(), inst.size(),   4, 4 ); 

    if(bnd)  bnd->save(dir,  "bnd.npy") ; 
    if(icdf) icdf->save(dir, "icdf.npy") ; 
}

void CSGFoundry::load( const char* dir_ )
{
    const char* dir = SPath::Resolve(dir_); 
    LOG(info) << dir ; 

    NP::ReadNames( dir, "name.txt", name );

    loadArray( solid , dir, "solid.npy" ); 
    loadArray( prim  , dir, "prim.npy" ); 
    loadArray( node  , dir, "node.npy" ); 
    loadArray( tran  , dir, "tran.npy" ); 
    loadArray( itra  , dir, "itra.npy" ); 
    loadArray( inst  , dir, "inst.npy" ); 
    loadArray( plan  , dir, "plan.npy" );  // often there are no planes in the geometry 

    bnd = NP::Load(dir, "bnd.npy"); 
    icdf = NP::Load(dir, "icdf.npy"); 
}


CSGFoundry*  CSGFoundry::Load(const char* dir) // static
{
    CSGFoundry* fd = new CSGFoundry();  
    fd->load(dir); 
    return fd ; 
} 

CSGFoundry*  CSGFoundry::Load(const char* base, const char* rel) // static
{
    CSGFoundry* fd = new CSGFoundry();  
    fd->load(base, rel); 
    return fd ; 
} 

void CSGFoundry::load( const char* base, const char* rel )
{
    std::stringstream ss ;   
    ss << base << "/" << rel ; 
    std::string dir = ss.str();   
    load( dir.c_str() ); 
}


template<typename T>
void CSGFoundry::loadArray( std::vector<T>& vec, const char* dir, const char* name )
{
    NP* a = NP::Load(dir, name);
    bool quiet = false ; 
    //if(a == nullptr && strcmp(name, "plan.npy") != 0)   // dont complain about lack of planes, many geometries do not have them 
    if(a == nullptr)   // dont complain about lack of planes, many geometries do not have them 
    { 
        LOG(LEVEL) << "FAIL for " << dir <<  "/" << name ; 
    }
    else
    { 
        assert( a->shape.size()  == 3 ); 
        unsigned ni = a->shape[0] ; 
        unsigned nj = a->shape[1] ; 
        unsigned nk = a->shape[2] ; 

        //std::cout  << " ni " << std::setw(5) << ni << " nj " << std::setw(1) << nj << " nk " << std::setw(1) << nk << " " << dir <<  "/" << name << std::endl ; 
        LOG(info) << " ni " << std::setw(5) << ni << " nj " << std::setw(1) << nj << " nk " << std::setw(1) << nk << " " << dir <<  "/" << name ; 
        //LOG(LEVEL) << " ni " << std::setw(5) << ni << " nj " << std::setw(1) << nj << " nk " << std::setw(1) << nk << " " << dir <<  "/" << name ; 

        vec.clear(); 
        vec.resize(ni); 
        memcpy( vec.data(),  a->bytes(), sizeof(T)*ni ); 
    }
}

template void CSGFoundry::loadArray( std::vector<CSGSolid>& , const char* , const char* ); 
template void CSGFoundry::loadArray( std::vector<CSGPrim>& , const char* , const char* ); 
template void CSGFoundry::loadArray( std::vector<CSGNode>& , const char* , const char* ); 
template void CSGFoundry::loadArray( std::vector<float4>& , const char* , const char* ); 
template void CSGFoundry::loadArray( std::vector<qat4>& , const char* , const char* ); 


void CSGFoundry::upload()
{
    inst_find_unique(); 
    LOG(info) << "[ " << desc() ; 
    assert( tran.size() == itra.size() ); 

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
void CSGFoundry::getInstanceTransformsGAS(std::vector<qat4>& select_inst, unsigned gas_idx ) const 
{
    qat4::select_instances_gas(inst, select_inst, gas_idx ) ;
}

const qat4* CSGFoundry::getInstanceGAS(unsigned gas_idx_ , unsigned ordinal)
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

MOI lookups Midx-mOrdinal-Iidx mesh/oridinal-of-mesh/instance-index of solid 

**/
void CSGFoundry::parseMOI(int& midx, int& mord, int& iidx, const char* moi) const 
{
    id->parseMOI(midx, mord, iidx, moi ); 
}
const char* CSGFoundry::getName(unsigned midx) const 
{
    return id->getName(midx); 
}

int CSGFoundry::getCenterExtent(float4& ce, int midx, int mord, int iidx) const 
{
    int rc = 0 ; 
    if( midx == -1 )
    {
        unsigned long long emm = 0ull ;   // hmm instance var ?
        iasCE(ce, emm); 
    }
    else
    {
        rc = target->getCenterExtent(ce, midx, mord, iidx);    // should use emm ?
    }
    return rc ; 
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






