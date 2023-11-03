
#include <vector>
#include <array>
#include <cstdlib>

#include "ssys.h"
#include "spath.h"

#include "SSim.hh"
#include "NP.hh"

#include "CSGFoundry.h"
#include "CSGQuery.h"
#include "CSGDraw.h"

#ifdef DEBUG_RECORD
#include "CSGRecord.h"
#endif

#include "CSGSimtraceRerun.h"
#include "SLOG.hh"

const plog::Severity CSGSimtraceRerun::LEVEL = SLOG::EnvLevel("CSGSimtraceRerun", "DEBUG") ; 


CSGSimtraceRerun::CSGSimtraceRerun()
    :
    sim(SSim::Create()),
    fd(CSGFoundry::Load()),      // via GEOM envvar 
    vv(fd ? fd->loadAux("Values/values.npy") : nullptr ),
    SELECTION(getenv("SELECTION")),
    selection(ssys::getenv_vec<int>("SELECTION",nullptr,',')),  // when no envvar gives fallback:nullptr  
    with_selection(selection && selection->size() > 0 ),  // SELECTION envvar must have values for with_selection:true 
    fold(spath::Resolve("$T_FOLD")),
    path0(spath::Join(fold, "simtrace.npy")),
    path1(spath::Join(fold, with_selection ? "simtrace_selection.npy" : "simtrace_rerun.npy" )),
    simtrace0(NP::LoadIfExists(path0)),
    simtrace1(with_selection ? NP::Make<float>(selection->size(),2,4,4) : NP::MakeLike(simtrace0)),
    qq0(simtrace0 ? (const quad4*)simtrace0->bytes() : nullptr),
    qq1(simtrace1 ? (quad4*)simtrace1->bytes() : nullptr),  
    q(fd ? new CSGQuery(fd) : nullptr),
    d(q  ? new CSGDraw(q,'Z') : nullptr)
{
    init(); 
}

void CSGSimtraceRerun::init()
{
    LOG_IF(fatal, fd == nullptr) << " NO GEOMETRY " ; 
    assert( fd ); 
    //LOG_IF(fatal, simtrace0 == nullptr) << " NO SIMTRACE INTERSECTS TO CHECK  " ; 
    //assert( simtrace0 ); 

    LOG(LEVEL) << ( d ? d->desc() : "-" ) ;
    LOG(LEVEL) << " fd.cfbase " << fd->cfbase ; 
    LOG(LEVEL) << " vv " << ( vv ? vv->sstr() : "-" ) ; 
    LOG(LEVEL) << "vv.lpath [" << ( vv->lpath.empty() ? "" : vv->lpath )  << "]" << std::endl << ( vv ? vv->descValues() : "-" ) ; 

    code_count.fill(0u); 
}
 
std::string CSGSimtraceRerun::desc() const
{
    std::stringstream ss ; 
    ss << "CSGSimtraceRerun::desc" << std::endl 
       << " fd " << ( fd ? "Y" : "N" ) << std::endl 
       << " fd.geom " << (( fd && fd->geom ) ? fd->geom : "-" ) << std::endl 
       << " " << CSGQuery::Label() << std::endl 
       << " path0 " << ( path0 ? path0 : "-" ) << std::endl 
       << " path1 " << ( path1 ? path1 : "-" ) << std::endl 
       << " simtrace0 " << ( simtrace0 ? simtrace0->sstr() : "-" ) << std::endl 
       << " simtrace1 " << ( simtrace1 ? simtrace1->sstr() : "-" ) << std::endl 
       << " SELECTION " << ( SELECTION ? SELECTION : "-" ) << std::endl 
       << " selection " << ( selection ? "Y" : "N" )
       << " selection.size " << ( selection ? selection->size() : -1 ) << std::endl 
       << " with_selection " << with_selection << std::endl 
       ;

    for(unsigned i=0 ; i < code_count.size() ; i++) ss << " code_count[" << i << "] " << code_count[i] << std::endl ; 
    std::string s = ss.str(); 
    return s ; 
}

std::string CSGSimtraceRerun::Desc(const quad4& isect1, const quad4& isect0) // static
{
    bool valid_isect0 = isect0.q0.f.w > isect0.q1.f.w ;   // dist > tmin
    bool valid_isect1 = isect1.q0.f.w > isect1.q1.f.w ;   // dist > tmin

    std::stringstream ss ; 
    ss << CSGQuery::Desc( isect0, "isect0", &valid_isect0 ) << std::endl ; 
    ss << CSGQuery::Desc( isect1, "isect1", &valid_isect1 ) << std::endl ; 
    std::string s = ss.str(); 
    return s ; 
}

unsigned  CSGSimtraceRerun::intersect_again(quad4& isect1, const quad4& isect0 )
{
    bool valid_isect = q->intersect_again(isect1, isect0); 
    bool valid_isect0 = isect0.q0.f.w > isect0.q1.f.w ;   // dist > tmin
    bool valid_isect1 = isect1.q0.f.w > isect1.q1.f.w ;   // dist > tmin
    unsigned code = ( unsigned(valid_isect0) << 1 ) | unsigned(valid_isect1) ;  
    assert( code < 4 ); 
    code_count[code] += 1 ; 
    code_count[4] += 1 ; 
    assert( valid_isect == valid_isect1 );  
    return code ; 
}

void CSGSimtraceRerun::intersect_again(unsigned idx, bool dump)
{
    const quad4& isect0 = qq0[idx] ; 
    quad4&       isect1 = qq1[idx] ;
    unsigned code = intersect_again(isect1, isect0); 

    if( dump || code == 1 || code == 2 )  
    {
        std::cout 
            << "CSGSimtraceRerun::intersect_again"
            << " idx " << std::setw(7) << idx 
            << " code " << code 
            << std::endl  
            << Desc(isect1, isect0) 
            << std::endl 
            ; 
    }
}

/**
CSGSimtraceRerun::intersect_again_selection
-----------------------------------------------

When a selection of indices is defined, by SELECTION envvar, save 
both isect0 and isect1 into the output simtrace_selection.npy array 

**/

void CSGSimtraceRerun::intersect_again_selection(unsigned i, bool dump)
{
    unsigned idx = selection->at(i) ;
    const quad4& isect0_ = qq0[idx] ;

    quad4&       isect0 = qq1[i*2+0] ;
    quad4&       isect1 = qq1[i*2+1] ;
    isect0 = isect0_ ; 

    unsigned code = intersect_again(isect1, isect0); 

    if( dump || code == 1 || code == 3 )
    {
        std::cout << " i " << std::setw(3) << i << " idx " << std::setw(7) << idx << " code " << code << std::endl ; 
        std::cout << Desc(isect1, isect0) << std::endl ; 
    }
}


void CSGSimtraceRerun::intersect_again()
{
    unsigned n = with_selection ? 
                                   ( selection ? selection->size() : 0u ) 
                                :  
                                    (simtrace0 ? simtrace0->shape[0] : 0u ) 
                                ;
 
    for(unsigned i=0 ; i < n ; i++) 
    {
        if( with_selection )
        {
             intersect_again_selection(i,true); 
        }
        else
        {
             //bool dump = i % 100000 == 0 ; 
             bool dump = false ; 
             intersect_again(i, dump); 
        }
    }
}

void CSGSimtraceRerun::save() const
{
    LOG_IF(error, simtrace1 == nullptr) << " simtrace1 null : cannot save " ; 
    if(simtrace1 == nullptr) return ; 

    LOG(info) << " path1 " << path1 ; 
    if(with_selection) simtrace1->set_meta<std::string>("SELECTION", SELECTION) ; 
    simtrace1->save(path1); 
}

void CSGSimtraceRerun::report() const 
{
    LOG(info) << "t.desc " << desc() ; 
#ifdef DEBUG_RECORD
    LOG(info) << "with : DEBUG_RECORD " ; 
    CSGRecord::Dump("CSGSimtraceRerun::report");   // HMM: that probably gets cleared for each intersect, so only the last set of CSGRecord gets dumped and saved 

    LOG(info) << " save CSGRecord.npy to fold " << fold ; 
    CSGRecord::Save(fold); 
#else
    std::cout << "not with DEBUG_RECORD : recompile with DEBUG_RECORD for full detailed recording " << std::endl ; 
#endif

#ifdef DEBUG
    std::cout << "with : DEBUG " << std::endl ; 
#else
    std::cout << "not with : DEBUG : recompile with DEBUG for full detailed recording " << std::endl ; 
#endif

}


