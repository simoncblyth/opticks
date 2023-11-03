
#include <vector>
#include <array>
#include <cstdlib>

#include "ssys.h"
#include "NP.hh"

#include "SSim.hh"

#include "CSGFoundry.h"
#include "CSGQuery.h"
#include "CSGDraw.h"

#include "CSGSimtraceSample.h"
#include "SLOG.hh"


const plog::Severity CSGSimtraceSample::LEVEL = SLOG::EnvLevel("CSGSimtraceSample", "DEBUG"); 

CSGSimtraceSample::CSGSimtraceSample()
    :
    sim(SSim::Create()),
    fd(CSGFoundry::Load()),      // via GEOM envvar 
    vv(fd ? fd->loadAux("Values/values.npy") : nullptr ),
    path(ssys::getenvvar("SAMPLE_PATH","/tmp/simtrace_sample.npy")),
    simtrace(NP::LoadIfExists(path)),
    qq(simtrace ? (quad4*)simtrace->bytes() : nullptr),  
    q(fd ? new CSGQuery(fd) : nullptr ),
    d(q  ? new CSGDraw(q,'Z') : nullptr)
{
    init(); 
}

void CSGSimtraceSample::init()
{
    LOG_IF( fatal, fd == nullptr ) << " fd NULL " ; 
    assert( fd ); 

    LOG(LEVEL) << " d(CSGDraw) " << ( d ? d->desc() : "-" )  ; 
    LOG(LEVEL) << " fd.cfbase " << ( fd ? fd->cfbase : "-" ) ; 
    LOG(LEVEL) << " vv " << ( vv ? vv->sstr() : "-" ) ; 
    LOG(LEVEL) << "vv.lpath [" << ( vv ? vv->lpath : "-" ) << "]"  ; 
    LOG(LEVEL) << "vv.descValues " << std::endl << ( vv ? vv->descValues() : "-" ) ; 
}
 
std::string CSGSimtraceSample::desc() const
{
    std::stringstream ss ; 
    ss << "CSGSimtraceSample::desc" << std::endl 
       << " fd " << ( fd ? "Y" : "N" ) << std::endl 
       << " fd.geom " << (( fd && fd->geom ) ? fd->geom : "-" ) << std::endl 
       << " " << CSGQuery::Label() << std::endl 
       << " path " << ( path ? path : "-" ) << std::endl 
       << " simtrace " << ( simtrace ? simtrace->sstr() : "-" ) << std::endl 
       ;

    std::string s = ss.str(); 
    return s ; 
}

int CSGSimtraceSample::intersect()
{
    LOG_IF(fatal, q == nullptr ) << " q NULL " ; 
    if(q == nullptr) return 0 ; 

    unsigned n = simtrace ? simtrace->shape[0] : 0u ; 
    int num_intersect = 0 ; 
    for(unsigned i=0 ; i < n ; i++) 
    {
        quad4& st = qq[i] ;
        bool valid_intersect = q->simtrace(st); 

        std::cout << CSGQuery::Desc(st, "-", &valid_intersect) << std::endl; 

        if(valid_intersect)
        {
            const float4& o = st.q2.f ; 
            const float4& v = st.q3.f ; 

            float t0 = -o.x/v.x ;  
            float z0 = o.z + t0*v.z ;  
            std::cout 
                << " o.x          " << std::setw(10) << std::fixed << std::setprecision(4) << o.x 
                << " v.x          " << std::setw(10) << std::fixed << std::setprecision(4) << v.x 
                << " t0(-o.x/v.x) " << std::setw(10) << std::fixed << std::setprecision(4) << t0 
                << " z0           " << std::setw(10) << std::fixed << std::setprecision(4) << z0 
                << std::endl
                ; 
        }

        if(valid_intersect) num_intersect += 1 ; 
    }

    LOG(LEVEL) << desc() << " n " << n << " num_intersect " << num_intersect ; 
    return num_intersect ; 
}


