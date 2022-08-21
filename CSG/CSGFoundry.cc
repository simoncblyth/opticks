#include <iostream>
#include <iomanip>
#include <array>
#include <vector>
#include <set>
#include <algorithm>
#include <cstring>
#include <csignal>
#include <cstdlib>

#include <glm/glm.hpp>
#include <glm/gtx/string_cast.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "SSys.hh"
#include "SSim.hh"
#include "SProc.hh"
#include "SStr.hh"
#include "SPath.hh"
#include "STime.hh"
#include "SBitSet.hh"
#include "SEventConfig.hh"
#include "SGeoConfig.hh"
#include "SOpticksResource.hh"
#include "NP.hh"

#include "SEvt.hh"
#include "SSim.hh"
#include "PLOG.hh"

#include "scuda.h"
#include "sqat4.h"
#include "sframe.h"


#include "OpticksCSG.h"
#include "CSGSolid.h"
#include "CU.h"
#include "CSGFoundry.h"
#include "SName.h"
#include "CSGTarget.h"
#include "CSGMaker.h"
#include "CSGCopy.h"

const unsigned CSGFoundry::IMAX = 50000 ; 

const plog::Severity CSGFoundry::LEVEL = PLOG::EnvLevel("CSGFoundry", "DEBUG" ); 
const int CSGFoundry::VERBOSE = SSys::getenvint("VERBOSE", 0); 

std::string CSGFoundry::descComp() const 
{
    std::stringstream ss ; 
    ss << "CSGFoundry::descComp"
       << " sim " << ( sim ? sim->desc() : "" )
       ;  
    std::string s = ss.str(); 
    return s ; 
}


void CSGFoundry::setPrimBoundary(unsigned primIdx, const char* bname ) 
{
    int bidx = sim->getBndIndex(bname); 
    assert( bidx > -1 ); 
    setPrimBoundary(primIdx, bidx); 
}

CSGFoundry::CSGFoundry()
    :
    d_prim(nullptr),
    d_node(nullptr),
    d_plan(nullptr),
    d_itra(nullptr),
    id(new SName(meshname)),   // SName takes a reference of the meshname vector of strings 
    target(new CSGTarget(this)),
    maker(new CSGMaker(this)),
    deepcopy_everynode_transform(true),
    last_added_solid(nullptr),
    last_added_prim(nullptr),
    sim(SSim::Get()),
    mtime(STime::EpochSeconds()),
    meta(),
    fold(nullptr),
    cfbase(nullptr),
    geom(nullptr),
    loaddir(nullptr),
    origin(nullptr),
    elv(nullptr)
{
    if(sim == nullptr) LOG(fatal) << "must SSim::Create before CSGFoundry::CSGFoundry " ; 
    assert(sim); 
    init(); 
}

/**
CSGFoundry::init
-----------------

Without sufficient reserved the vectors may reallocate on any push_back invalidating prior pointers.
Yes, but generally indices are used rather than pointers to avoid this kinda issue. 

**/
void CSGFoundry::init()
{
    solid.reserve(IMAX); 
    prim.reserve(IMAX); 
    node.reserve(IMAX); 
    plan.reserve(IMAX); 
    tran.reserve(IMAX); 
    itra.reserve(IMAX); 
    inst.reserve(IMAX); // inst added July 2022 

    setMeta(); 
}

void CSGFoundry::setMeta()
{
    setMeta<std::string>("creator", SProc::ExecutableName() ); 
    setMeta<int>("mtime", mtime); 

    const char* vars = R"( 
HOME
USER
SCRIPT
PWD
CMDLINE
)" ; 
    // BASH_SOURCE fails to export  
    std::stringstream ss;  
    ss.str(vars)  ;
    std::string s;
    while (std::getline(ss, s, '\n')) 
    {   
        if(s.empty()) continue ; 
        const char* k = s.c_str(); 
        const char* v = getenv(k) ; 
        if(v) setMeta<std::string>(k, v); 
        LOG(LEVEL) << std::setw(20) << k << " : " << ( v ? v : "-" ) ;   
    }   
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
       << " gas " << gas.size()
       /*
       << " ins " << ins.size()
       << " sensor_identifier " << sensor_identifier.size()
       << " sensor_index " << sensor_index.size()
       */
       << " meshname " << meshname.size()
       << " mmlabel " << mmlabel.size()
       << " mtime " << mtime
       << " mtimestamp " << STime::Format(mtime)
       << " sim " << ( sim ? "Y" : "N" )
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


unsigned CSGFoundry::getNumMeshes() const 
{
    return meshname.size() ; 
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
    dst->mtime = src->mtime ; 
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


/**
CSGFoundry::getPrimName
------------------------

After CSGFoundry.py:make_primIdx_meshname_dict
see notes/issues/cxs_2d_plotting_labels_suggest_meshname_order_inconsistency.rst

**/

void CSGFoundry::getPrimName( std::vector<std::string>& pname ) const 
{
    for(unsigned i=0 ; i < prim.size() ; i++)
    {
        const CSGPrim& pr = prim[i] ; 
        unsigned midx = pr.meshIdx(); 
        const std::string& mname = getMeshName(midx); 
        LOG(debug) << " primIdx " << std::setw(4) << i << " midx " << midx << " mname " << mname  ;  
        pname.push_back(mname);  
    }
}




const char* CSGFoundry::getMeshName(unsigned midx) const 
{
    assert( midx < meshname.size() ); 
    return meshname[midx].c_str() ; 
}






/**
CSGFoundry::findMeshIndex
--------------------------

SName::findIndex uses "name starts with query string" matching
so names like HamamatsuR12860sMask_virtual0x5f50520
can be matched without the pointer suffix. 

**/

int CSGFoundry::findMeshIndex(const char* qname) const 
{
    unsigned count = 0 ; 
    int max_count = 1 ; 
    int midx = id->findIndex(qname, count, max_count); 
    return midx ; 
}

/**
CSGFoundry::getMeshIndexWithName
----------------------------------

SGeo 

**/

int CSGFoundry::getMeshIndexWithName(const char* qname, bool startswith) const 
{
    return id->findIndexWithName(qname, startswith) ; 
}


int CSGFoundry::lookup_mtline(int mtindex) const 
{
    return sim->lookup_mtline(mtindex) ;  
}



/**
CSGFoundry::descELV
----------------------

TODO: move elsewhwre, as it all can be done with SBitSet and SName instances 

**/

const std::string CSGFoundry::descELV(const SBitSet* elv) const 
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
        int midx = findMeshIndex(mn.c_str()); 
        //assert( int(p) == midx );  

        ss 
            << std::setw(3) << p << ":" 
            << std::setw(3) << midx << ":" 
            << mn << std::endl 
            ;  
    }

    ss << "EXCLUDE:" << exclude_pos.size() << std::endl << std::endl ;  
    for(unsigned i=0 ; i < exclude_pos.size() ; i++)
    {
        const unsigned& p = exclude_pos[i] ; 
        const std::string& mn = getMeshName(p) ; 
        int midx = findMeshIndex(mn.c_str()); 
        //assert( int(p) == midx );  

        ss 
           << std::setw(3) << p << ":" 
           << std::setw(3) << midx << ":" 
           << mn << std::endl 
           ;  
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
    mismatch += CompareVec( "gas"  , a->gas , b->gas ); 
    /* 
    mismatch += CompareVec( "ins"  , a->ins , b->ins ); 
    mismatch += CompareVec( "sensor_identifier"  , a->sensor_identifier , b->sensor_identifier ); 
    mismatch += CompareVec( "sensor_index"       , a->sensor_index      , b->sensor_index ); 
    */
    if( mismatch != 0 ) LOG(fatal) << " mismatch FAIL ";  
    //assert( mismatch == 0 ); 

    mismatch += SSim::Compare( a->sim, b->sim, true ); 

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

std::string CSGFoundry::descInstance() const
{
    std::vector<int>* idxs = SSys::getenvintvec("IDX", ',');
    std::stringstream ss ; 
    if(idxs == nullptr)
    {
        ss << " no IDX " << std::endl ; 
    }
    else
    {
        for(unsigned i=0 ; i < idxs->size() ; i++)
        {
            int idx = (*idxs)[i] ; 
            ss << descInstance(idx) ;          
        }
    } 
    std::string s = ss.str(); 
    return s ; 
}

/**
CSGFoundry::descInstance
---------------------------

::

    c ; IDX=0,10,100 METH=descInstance ./CSGTargetTest.sh remote 
  

**/

std::string CSGFoundry::descInstance(unsigned idx) const
{
    std::stringstream ss ; 
    ss << "CSGFoundry::descInstance"
       << " idx " << std::setw(7) << idx 
       << " inst.size " << std::setw(7) << inst.size()
        ; 

    if(idx >= inst.size() ) 
    {
        ss << " idx OUT OF RANGE " ; 
    }
    else
    {
        const qat4& q = inst[idx] ;      
        int ins_idx,  gas_idx, sensor_identifier, sensor_index ;
        q.getIdentity(ins_idx,  gas_idx, sensor_identifier, sensor_index );


        const CSGSolid* so = getSolid(gas_idx); 
 
        ss << " idx " << std::setw(7) << idx               
           << " ins " << std::setw(5) << ins_idx                
           << " gas " << std::setw(2) << gas_idx                
           << " s_ident " << std::setw(7) << sensor_identifier
           << " s_index " << std::setw(5) << sensor_index       
           << " so " << so->desc()
           ;
    }
    ss << std::endl ; 
    std::string s = ss.str(); 
    return s ; 
}


std::string CSGFoundry::descInst(unsigned ias_idx_, unsigned long long emm ) const
{
    std::stringstream ss ; 
    for(unsigned i=0 ; i < inst.size() ; i++)
    {
        const qat4& q = inst[i] ;      
        int ins_idx,  gas_idx, sensor_identifier, sensor_index ;
        q.getIdentity(ins_idx,  gas_idx, sensor_identifier, sensor_index );

        bool gas_enabled = emm == 0ull ? true : ( emm & (0x1ull << gas_idx)) ;  
        if( gas_enabled )
        {
            const CSGSolid* so = getSolid(gas_idx); 
            ss 
                << " i " << std::setw(5) << i               
                << " ins " << std::setw(5) << ins_idx                
                << " gas " << std::setw(2) << gas_idx                
                << " s_identifier " << std::setw(7) << sensor_identifier
                << " s_index " << std::setw(5) << sensor_index
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

        int ins_idx,  gas_idx, sensor_identifier, sensor_index ;
        q.getIdentity(ins_idx,  gas_idx, sensor_identifier, sensor_index );


        bool gas_enabled = emm == 0ull ? true : ( emm & (0x1ull << gas_idx)) ;  
        if( gas_enabled )
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

Q: why the awkward external solid_selection vector ?

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
            unsigned count = std::count(ss.begin(), ss.end(), j );  // count if j is already collected 
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
    const char* bndName = sim ? sim->getBndName(boundary) : "-bd" ;

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

Used for example from 

1. CSG_GGeo_Convert::addInstances when creating CSGFoundry from GGeo
2. CSGCopy::copy/CSGCopy::copySolidInstances when copy a loaded CSGFoundry to apply a selection

**/

void CSGFoundry::addInstance(const float* tr16, int gas_idx, int sensor_identifier, int sensor_index )
{
    qat4 instance(tr16) ;  // identity matrix if tr16 is nullptr 
    int ins_idx = int(inst.size()) ;

    instance.setIdentity( ins_idx, gas_idx, sensor_identifier, sensor_index );

    LOG(debug) 
        << " ins_idx " << ins_idx 
        << " gas_idx " << gas_idx 
        << " sensor_identifier " << sensor_identifier
        << " sensor_index " << sensor_index
        ;

    inst.push_back( instance );
} 

void CSGFoundry::addInstancePlaceholder()
{
    const float* tr16 = nullptr ; 
    int gas_idx = -1 ; 
    int sensor_identifier = -1 ; 
    int sensor_index = -1 ;  

    addInstance(tr16, gas_idx, sensor_identifier, sensor_index );  
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


const char* CSGFoundry::BASE = "$DefaultOutputDir" ; // incorporates GEOM if defined
const char* CSGFoundry::RELDIR = "CSGFoundry"  ;

/**
CSGFoundry::getBaseDir
-------------------------

Returns value of CFBASE envvar if defined, otherwise resolves '$DefaultOutputDir' which 
is for example /tmp/$USER/opticks/$GEOM/SProc::ExecutableName

**/

const char* CSGFoundry::getBaseDir(bool create) const
{
    const char* cfbase_default = SPath::Resolve(BASE, create ? DIRPATH : NOOP );  //   
    const char* cfbase = SSys::getenvvar("CFBASE", cfbase_default );  
    return cfbase ? strdup(cfbase) : nullptr ; 
}

void CSGFoundry::save() const 
{
    const char* cfbase = getBaseDir(true) ; 
    if( cfbase == nullptr )
    {
        LOG(fatal) << "cannot save unless CFBASE envvar defined or geom has been set " ; 
        return ;   
    }
    save(cfbase, RELDIR );  
}


const char* cfdir = SPath::Resolve("$DefaultOutputDir", DIRPATH);


void CSGFoundry::save(const char* base, const char* rel) const 
{
    if(rel == nullptr) rel = RELDIR ; 
    std::stringstream ss ;   
    ss << base << "/" << rel ; 
    std::string dir = ss.str();   
    save_(dir.c_str()); 
}

/**
CSGFoundry::saveAlt
-----------------------

Write geometry to $CFBaseAlt/CSGFoundry currently used as workaround so 
that the python view of dynamic prim selection geometry can match 
the actual uploaded geometry. 

See notes/issues/primIdx-and-skips.rst
**/

void CSGFoundry::saveAlt() const 
{
    const char* cfbase_alt = SOpticksResource::CFBaseAlt(); 
    if( cfbase && cfbase_alt && strcmp(cfbase, cfbase_alt) == 0 )
    {
        LOG(fatal) 
            << "cannot saveAlt as cfbase_alt directory matched the loaded directory "
            << " cfbase " << cfbase
            << " cfbase_alt " << cfbase_alt
            ;
    } 
    else
    {
        LOG(info) << " cfbase " << cfbase << " cfbase_alt " << cfbase_alt ; 
        save(cfbase_alt, RELDIR); 
    }
}


/**
CSGFoundry::save_
------------------

Have observed that whilst changing geometry this can lead to "mixed" exports 
with the contents of CSGFoundry directory containing arrays from multiple exports. 
The inconsistency causes crashes.  

TODO: find way to avoid this, by deleting the folder ahead : or asserting on consistent time stamps
on loading 
 
**/
void CSGFoundry::save_(const char* dir_) const 
{
    const char* dir = SPath::Resolve(dir_, DIRPATH); 
    LOG(info) << dir ; 

    if(meshname.size() > 0 ) NP::WriteNames( dir, "meshname.txt", meshname );

    std::vector<std::string> primname ; 
    getPrimName(primname); 
    if(primname.size() > 0 ) NP::WriteNames( dir, "primname.txt", primname );

    if(mmlabel.size() > 0 )  NP::WriteNames( dir, "mmlabel.txt", mmlabel );
    if(hasMeta())  NP::WriteString( dir, "meta.txt", meta.c_str() ); 
              
    if(solid.size() > 0 ) NP::Write(dir, "solid.npy",  (int*)solid.data(),  solid.size(), 3, 4 ); 
    if(prim.size() > 0 ) NP::Write(dir, "prim.npy",   (float*)prim.data(), prim.size(),   4, 4 ); 
    if(node.size() > 0 ) NP::Write(dir, "node.npy",   (float*)node.data(), node.size(),   4, 4 ); 
    if(plan.size() > 0 ) NP::Write(dir, "plan.npy",   (float*)plan.data(), plan.size(),   1, 4 ); 
    if(tran.size() > 0 ) NP::Write(dir, "tran.npy",   (float*)tran.data(), tran.size(),   4, 4 ); 
    if(itra.size() > 0 ) NP::Write(dir, "itra.npy",   (float*)itra.data(), itra.size(),   4, 4 ); 
    if(inst.size() > 0 ) NP::Write(dir, "inst.npy",   (float*)inst.data(), inst.size(),   4, 4 ); 

    if(sim)
    {
        LOG(LEVEL) << " SSim::save " << dir ;  
        sim->save(dir, SSim::RELDIR );  
    }
    else
    {
        LOG(LEVEL) << " CANNOT SSim::save AS sim null " ;  
    }
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



const char* CSGFoundry::load_FAIL_base_null_NOTES = R"(
CSGFoundry::load_FAIL_base_null_NOTES
======================================

Loading of the geometry failed as the base directory was not provided. 
This typically means you are attempting to run an executable directly 
when you should be running the script that sets up the needed environment
that specifies which geometry directory to load. 

This error can also happen when attempting to load event+geometry
that was previously written to directories below /tmp 
but has subsequently been "cleaned" living the directory structure
but with all the directories emptied. 

Relevant envvars : CFBASE and GEOM 

)" ; 


void CSGFoundry::load(const char* base_, const char* rel) 
{
    if(base_ == nullptr) LOG(error) << load_FAIL_base_null_NOTES ; 
    assert(base_); 
    bool conventional = strcmp( rel, RELDIR) == 0  ; 
    if(!conventional) LOG(fatal) << "Convention is for rel to be named [" << RELDIR << "] not: [" << rel << "]"  ; 
    assert(conventional); 

    const char* base = SPath::Resolve(base_, NOOP); 
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
    // HMM: maybe just pass the pointer, as keep forgetting about this 
    // NOPE: that would be wrong need to save into a new cfbase 
    // for consistency between results and geometry 

    LOG(LEVEL) << " CAUTION HOW YOU USE THIS : MISUSE CAN EASILY LEAD TO INCONSISTENCY BETWEEN RESULTS AND GEOMETRY " ; 
    return origin ? origin->cfbase : cfbase ; 
}



std::string CSGFoundry::descBase() const 
{
    const char* cfb = getCFBase(); 
    const char* ocfb = getOriginCFBase(); 
    std::stringstream ss ; 
    ss << "CSGFoundry.descBase " << std::endl 
       << " CFBase       " << ( cfb ? cfb : "-" ) << std::endl 
       << " OriginCFBase " << ( ocfb ? ocfb : "-" ) << std::endl 
       ;
    return ss.str(); 
}



/**
CSGFoundry::load
------------------

**/


const char* CSGFoundry::LOAD_FAIL_NOTES = R"LITERAL(
CSGFoundry::LOAD_FAIL_NOTES
==============================

The CSGFoundry directory does not exist. To create it you probably need to 
run one of several CSGFoundry creating scripts. Which one to use depends on 
what the geometry is that you want to create. Some of the scripts require 
setting the GEOM name within $HOME/.opticks/GEOM.txt to pick between 
different geometries. 

CSG/CSGMakerTest.sh 
    CSG level creation of simple test CSGFoundry 

    NB somewhat misleadinging this saves to directories such as::

        /tmp/$USER/opticks/GeoChain/BoxedSphere/CSGFoundry/


GeoChain/translate.sh
    translation of Geant4 defined test geometry 

CSG_GGeo/run.sh 
    translation of OPTICKS_KEY identified GGeo geometry into CSGFoundry


)LITERAL" ; 


void CSGFoundry::load( const char* dir_ )
{
    const char* dir = SPath::Resolve(dir_, NOOP ); 
    bool readable = SPath::IsReadable(dir); 
    if( readable == false )
    {
        LOG(fatal) << " dir is not readable [" << dir << "]" ;
        std::cout << LOAD_FAIL_NOTES << std::endl ; 
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

    sim = NP::Exists(dir, "SSim") ? SSim::Load(dir, SSim::RELDIR ) : nullptr ; 
    mtime = MTime(dir, "solid.npy"); 
}


/**
CSGFoundry::MTime
-------------------

Use STime::Format(mtime) to convert the int into a timestamp string

**/
int CSGFoundry::MTime(const char* dir, const char* fname_) // static
{
    const char* fname = fname_ ? fname_ : "solid.npy" ; 
    const char* path = SPath::Resolve(dir, fname, NOOP); 
    return SPath::mtime(path); 
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
CSGFoundry::ELVString
-----------------------

String configuring dynamic shape selection of form : t110,117,134 or null when 
there is no selection.  The value is obtained from either:

1. SGeoConfig::ELVSelection() which defaults to the OPTICKS_ELV_SELECTION envvar value 
   and can be changed by the SGeoConfig::SetELVSelection static, a comma delimited list of 
   mesh names is expected, for example: 
   "NNVTMCPPMTsMask_virtual0x,HamamatsuR12860sMask_virtual0x,mask_PMT_20inch_vetosMask_virtual0x"

   If any of the names are not found in the geometry the selection request is ignored.   

2. ELV envvar which is expected to be a comma delimited list of mesh indices, eg "t110,117,134"


HMM: could detect the indices or names form in use and parse accordingly, then would only need one envvar
or simply remove the lower level ELV form or obfuscate it with a long envvar key 

**/

const char* CSGFoundry::ELVString(const SName* id)
{
    const char* elv_selection_ = SGeoConfig::ELVSelection() ; 
    const char* elv = nullptr ; 
    if( elv_selection_ )
    {
        bool has_names = id->hasNames(elv_selection_);  
        if(has_names)
        {
            elv = id->getIDXListFromNames(elv_selection_, ',', "t" ); 
        }
        else
        {
            LOG(fatal) << "geometry does not have all the elv_selection_ names [" << elv_selection_ << "] NO SELECTION WILL BE APPLIED " ; 
        }
    }
    else
    {
        elv = SSys::getenvvar("ELV", nullptr ); 
    }

    LOG(LEVEL) 
        << " elv_selection_ " << elv_selection_
        << " elv " << elv
        ; 

    return elv ; 
}

const SBitSet* CSGFoundry::ELV(const SName* id)
{
    unsigned num_meshname = id->getNumName(); 
    const char* elv_ = ELVString(id); 
    SBitSet* elv = elv_ ? SBitSet::Create(num_meshname, elv_ ) : nullptr ; 
    return elv ; 
}


/**
CSGFoundry::Load
-------------------

This argumentless Load method is special, unlike other methods 
it provides dynamic prim selection based in the ELV envvar which uses
CSGFoundry::CopySelect to dynamically create a CSGFoundry based
on the elv SBitSet

Dynamic prim selection (using ELV) without saving the CSGFoundry of the modified geometry 
can be useful for render performance scanning to find geometry bottlenecks 
but it is just not appropriate when wishing to run multiple executables over the same geometry 
and do detailed analysis of the results. In this situation it is vital to have a more constant 
CSGFoundry geometry folder that is read by multiple executables including rendering and 
python analysis machinery. 

**/

bool CSGFoundry::Load_saveAlt = SSys::getenvbool("CSGFoundry_Load_saveAlt") ; 

CSGFoundry* CSGFoundry::Load() // static
{
    LOG(LEVEL) << " argumentless " ; 
    CSGFoundry* src = CSGFoundry::Load_() ; 
    if(src == nullptr) return nullptr ; 

    SGeoConfig::GeometrySpecificSetup(src->id); 

    const SBitSet* elv = ELV(src->id); 
    CSGFoundry* dst = elv ? CSGFoundry::CopySelect(src, elv) : src  ; 

    if( elv != nullptr && Load_saveAlt)
    {
        LOG(error) << " non-standard dynamic selection CSGFoundry_Load_saveAlt " ; 
        dst->saveAlt() ; 
    }


   


    return dst ; 
}

CSGFoundry* CSGFoundry::CopySelect(const CSGFoundry* src, const SBitSet* elv )
{
    assert(elv);
    LOG(info) << elv->desc() << std::endl << src->descELV(elv) ; 
    CSGFoundry* dst = CSGCopy::Select(src, elv ); 
    dst->setOrigin(src); 
    dst->setElv(elv); 
    dst->setOverrideSim(src->sim);   
    // pass the SSim pointer from the loaded src instance, 
    // overriding the empty dst SSim instance 
    return dst ; 
}



/**
CSGFoundry::ResolveCFBase
---------------------------

The cfbase path is sensitive to envvars CFBASE, OPTICKS_KEY, OPTICKS_GEOCACHE_PREFIX 

Load precedence:

0. when GEOM and "GEOM"_CFBaseFromGEOM are defined that directory is used
1. when CFBASE envvar is defined the CSGFoundry directory within CFBASE dir will be loaded
2. otherwise SOpticksResource::CFBase will invoke CGDir to obtain the CSG_GGeo directory 
   corresponding to the current OPTICKS_KEY envvar 

**/

const char* CSGFoundry::ResolveCFBase_() // static 
{
    const char* cfbase = nullptr ; 
    cfbase = SOpticksResource::CFBaseFromGEOM() ;
    if(cfbase == nullptr) cfbase = SOpticksResource::CFBase() ;  // standard or override
    return cfbase ; 
}

const char* CSGFoundry::ResolveCFBase() // static 
{
    const char* cfbase = ResolveCFBase_(); 
    bool readable = cfbase == nullptr ? false : SPath::IsReadable(cfbase, "CSGFoundry") ; 
    LOG(LEVEL) << " cfbase " << cfbase << " readable " << readable ; 

    if(!readable)
        LOG(fatal) << " cfbase/CSGFoundy directory " << cfbase << "/CSGFoundry" << " IS NOT READABLE " ; 
   
    return readable ? cfbase : nullptr ; 
}

/**
CSGFoundry::Load_
-------------------
**/

CSGFoundry* CSGFoundry::Load_() // static
{
    const char* cfbase = ResolveCFBase() ; 
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


    CSGFoundry* fd = new CSGFoundry();  
    fd->load(base, rel); 
    return fd ; 
} 


void CSGFoundry::setOverrideSim( const SSim* override_sim )
{
    sim = override_sim ; 
}
const SSim* CSGFoundry::getSim() const 
{
    return sim ; 
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

This currently taking 20s for full JUNO, where total runtime for one event is 24s. 

**/

void CSGFoundry::upload()
{ 
    
    LOG(LEVEL) << "[ inst_find_unique " ; 
    inst_find_unique(); 
    LOG(LEVEL) << "] inst_find_unique " ; 

    LOG(LEVEL) << desc() ; 

    assert( tran.size() == itra.size() ); 

    bool is_uploaded_0 = isUploaded(); 
    if(is_uploaded_0) LOG(fatal) << "HAVE ALREADY UPLOADED : THIS CANNOT BE DONE MORE THAN ONCE " ; 
    assert(is_uploaded_0 == false); 

    // allocates and copies
    LOG(LEVEL) << "[ CU::UploadArray "  ; 
    d_prim = prim.size() > 0 ? CU::UploadArray<CSGPrim>(prim.data(), prim.size() ) : nullptr ; 
    d_node = node.size() > 0 ? CU::UploadArray<CSGNode>(node.data(), node.size() ) : nullptr ; 
    d_plan = plan.size() > 0 ? CU::UploadArray<float4>(plan.data(), plan.size() ) : nullptr ; 
    d_itra = itra.size() > 0 ? CU::UploadArray<qat4>(itra.data(), itra.size() ) : nullptr ; 
    LOG(LEVEL) << "] CU::UploadArray "  ; 

    bool is_uploaded_1 = isUploaded(); 
    if(!is_uploaded_1) LOG(fatal) << "FAILED TO UPLOAD" ; 
    assert(is_uploaded_1 == true); 

    //const char* ocfb = getOriginCFBase(); 
    //   actually that means results will be placed in folder inconsistent with the geometry 

    const char* cfb = getCFBase(); 
    SGeo::SetLastUploadCFBase(cfb) ; 
    LOG(LEVEL) << "]" ; 
}

bool CSGFoundry::isUploaded() const
{
    return d_prim != nullptr && d_node != nullptr ; 
}



void CSGFoundry::inst_find_unique()
{
    qat4::find_unique_gas( inst, gas ); 
    //qat4::find_unique( inst, ins, gas, sensor_identifier, sensor_index ); 
}

unsigned CSGFoundry::getNumUniqueGAS() const
{
    return gas.size(); 
}
unsigned CSGFoundry::getNumUniqueIAS() const
{
    return 1 ; 
}

/*
unsigned CSGFoundry::getNumUniqueINS() const
{
    return ins.size(); 
}
*/



unsigned CSGFoundry::getNumInstancesIAS(int ias_idx, unsigned long long emm) const
{
    return qat4::count_ias(inst, ias_idx, emm );  
}
void CSGFoundry::getInstanceTransformsIAS(std::vector<qat4>& select_inst, int ias_idx, unsigned long long emm ) const 
{
    qat4::select_instances_ias(inst, select_inst, ias_idx, emm ) ;
}


unsigned CSGFoundry::getNumInstancesGAS(int gas_idx) const
{
    return qat4::count_gas(inst, gas_idx );  
}

void CSGFoundry::getInstanceTransformsGAS(std::vector<qat4>& select_qv, int gas_idx ) const 
{
    qat4::select_instances_gas(inst, select_qv, gas_idx ) ;
}

void CSGFoundry::getInstancePointersGAS(std::vector<const qat4*>& select_qi, int gas_idx ) const 
{
    qat4::select_instance_pointers_gas(inst, select_qi, gas_idx ) ;
}

/**
CSGFoundry::getInstanceIndex
------------------------------

Returns index of the ordinal-th instance with the provided gas_idx or -1 if not found
Note that the gas ordinal is confusingly referred to as the iidx in many places.
BUT that iidx is not the global instance index it is the index 
within occurences of the gas_idx_ GAS/compositeSolid

**/
int CSGFoundry::getInstanceIndex(int gas_idx_ , unsigned ordinal) const 
{
    return qat4::find_instance_gas(inst, gas_idx_, ordinal);
}

/**
CSGFoundry::getInstanceGAS
----------------------------

Instance transform using gas local ordinal (not the global instance index).

NB the method *CSGFoundry::getInst* provides the instance transform 
from the global instance index argument. 

**/

const qat4* CSGFoundry::getInstanceGAS(int gas_idx_ , unsigned ordinal) const 
{
    int index = getInstanceIndex(gas_idx_, ordinal); 
    return index > -1 ? &inst[index] : nullptr ; 
}

std::string CSGFoundry::descGAS() const 
{
    std::stringstream ss ; 
    ss << desc() << std::endl ; 
    for(unsigned i=0 ; i < gas.size() ; i++)
    {   
        int gas_idx = gas[i]; 
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
    id->parseMOI(midx, mord, iidx, moi );  // SName::parseMOI
}
const char* CSGFoundry::getName(unsigned midx) const 
{
    return id->getName(midx); 
}


/**
CSGFoundry::getFrame
---------------------

Replacing most of CSGOptiX::setComposition

**/
const char* CSGFoundry::FRS = "-1" ; 

const char* CSGFoundry::getFrame_NOTES = R"(
CSGFoundry::getFrame_NOTES
===========================

When CSGFoundry::getFrame fails due to the MOI/FRS string used to target 
a volume of the geometry failing to find the targetted volume 
it is usually due to the spec not being appropriate for the geometry. 

When using U4VolumeMaker it is sometimes possible to 
debug the bad specification string by rerunning with the below 
envvars set in order to dump PV/LV/SO names from the full and sub trees.::

   export U4VolumeMaker_PVG_WriteNames=1
   export U4VolumeMaker_PVG_WriteNames_Sub=1

This writes the names for the full volume tree to eg::

   /tmp/$USER/opticks/U4VolumeMaker_PVG_WriteNames 
   /tmp/$USER/opticks/U4VolumeMaker_PVG_WriteNames_Sub 

Grab these from remote with::

   u4
   ./U4VolumeMaker.sh grab 


)" ; 


sframe CSGFoundry::getFrame() const 
{
    const char* moi_or_iidx = SSys::getenvvar("MOI",FRS);   // TODO: MOI->FRS perhaps ?
    return getFrame(moi_or_iidx); 
}
sframe CSGFoundry::getFrame(const char* frs) const 
{
    sframe fr ; 
    int rc = getFrame(fr, frs ? frs : FRS );
    if(rc != 0) LOG(error) << " frs " << frs << std::endl << getFrame_NOTES ; 
    if(rc != 0) std::raise(SIGINT); 

    fr.prepare();  // creates Tran<double>
    return fr ; 
}



/**
CSGFoundry::getFrame
---------------------

frs
    frame specification string is regarded to "looks_like_moi" when 
    it starts with a letter or it contains ":" or it is "-1". 
    For such strings parseMOI is used to extract midx, mord, oodx
 
    Otherwise the string is assumed to be inst_idx and iidxg
    parsed as an integer 


Q: is indexing by MOI and inst_idx equivalent ? OR: Can a MOI be converted into inst_idx and vice versa ?

* NO not for global prims : for example for all the repeated prims in the global geometry 
  there is only one inst_idx (zero) but there are many possible MOI  

* NO not for most instanced prim, where all the prim within an instance 
  share the same inst_idx and transform

* BUT for the outer prim of an instance a correspondence is possible  

**/



int CSGFoundry::getFrame(sframe& fr, const char* frs ) const 
{
    int rc = 0 ; 
    bool looks_like_moi = SStr::StartsWithLetterAZaz(frs) || strstr(frs, ":") || strcmp(frs,"-1") == 0 ; 
    if(looks_like_moi)
    {
        int midx, mord, iidx ;  // mesh-index, mesh-ordinal, gas-instance-index
        parseMOI(midx, mord, iidx,  frs );  
        rc = getFrame(fr, midx, mord, iidx); 
    }
    else
    {
         int inst_idx = SName::ParseIntString(frs, 0) ; 
         rc = getFrame(fr, inst_idx); 
    }

    fr.set_propagate_epsilon( SEventConfig::PropagateEpsilon() ); 
    fr.frs = strdup(frs); 
    LOG(LEVEL) << " fr " << fr ;    // no grid has been set at this stage, just ce,m2w,w2m
    if(rc != 0) LOG(error) << "Failed to lookup frame with frs [" << frs << "] looks_like_moi " << looks_like_moi  ;   
    return rc ; 
}

int CSGFoundry::getFrame(sframe& fr, int inst_idx) const
{
    return target->getFrame( fr, inst_idx ); 
}



int CSGFoundry::getFrame(sframe& fr, int midx, int mord, int iidxg) const 
{
    int rc = 0 ; 
    if( midx == -1 )
    { 
        unsigned long long emm = 0ull ;   // hmm instance var ?
        iasCE(fr.ce, emm); 
    }
    else
    {
        rc = target->getFrame( fr, midx, mord, iidxg );  
    }
    return rc ; 
}







/**
TODO CSGFoundry::flatIdxToMOI ?
---------------------------------

Method to convert the flat global inst_idx into a more informative MOI mname:mord:gas_iidx
that is likely to last longer before changing its meaning 
(although there can be several CSGPrim within the GAS inst_idx could simply use the first 
which should be the outer one)

**/


/**
CSGFoundry::getCenterExtent
-------------------------------

For midx -1 returns ce obtained from the ias bbox, 
otherwise uses CSGTarget to lookup the center extent. 

For global geometry which typically means a default iidx of 0 
there is special handling for iidx -1/-2/-3 in CSGTarget::getCenterExtent

iidxg -1
    uses getLocalCenterExtent

iidxg -2
    uses SCenterExtentFrame xyzw : ordinary XYZ frame 

iidxg -3
    uses SCenterExtentFrame rtpw : tangential RTP frame 


NB iidxg is the gas ordinal index, that is NOT the global instance index 

**/

int CSGFoundry::getCenterExtent(float4& ce, int midx, int mord, int iidxg, qat4* m2w, qat4* w2m  ) const 
{
    int rc = 0 ; 
    if( midx == -1 )
    { 
        unsigned long long emm = 0ull ;   // hmm instance var ?
        iasCE(ce, emm); 
    }
    else
    {
        rc = target->getCenterExtent(ce, midx, mord, iidxg, m2w, w2m );   
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


