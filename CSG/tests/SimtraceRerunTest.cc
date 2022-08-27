/**
SimtraceRerunTest.cc
================================

Roughly based on CSGQueryTest.cc but updated for simtrace array reruns

TODO: 

* shoot ray, trying to duplicate the simtrace item

**/

#include <array>

#include "SSys.hh"
#include "SSim.hh"
#include "SPath.hh"
#include "NP.hh"
#include "OPTICKS_LOG.hh"

#include "CSGFoundry.h"
#include "CSGQuery.h"
#include "CSGDraw.h"

struct SimtraceRerunTest
{ 
    SSim* sim ; 
    const CSGFoundry* fd ; 

    const char* path0 ; 
    const char* path1 ; 
    NP* simtrace0 ; 
    NP* simtrace1 ; 

    const CSGQuery* q ; 
    CSGDraw* d ; 

    std::array<unsigned, 5> code_count ; 

    SimtraceRerunTest(); 
    void init(); 
    std::string desc() const ; 

    void intersect_again(unsigned idx, bool dump); 
    void intersect_again(); 
    void save() const ; 

};

inline SimtraceRerunTest::SimtraceRerunTest()
    :
    sim(SSim::Create()),
    fd(CSGFoundry::Load()),
    path0(SPath::Resolve("$T_FOLD/simtrace.npy", NOOP)),
    path1(SPath::Resolve("$T_FOLD/simtrace_rerun.npy", NOOP)),
    simtrace0(NP::Load(path0)),
    simtrace1(NP::MakeLike(simtrace0)),
    q(new CSGQuery(fd)),
    d(new CSGDraw(q,'Z'))
{
    init(); 
}

inline void SimtraceRerunTest::init()
{
    LOG(info) << " fd.geom " << fd->geom ; 
    d->draw("SimtraceRerunTest");

    code_count.fill(0u); 
}
 
inline std::string SimtraceRerunTest::desc() const
{
    std::stringstream ss ; 
    ss << "SimtraceRerunTest::desc" << std::endl 
       << " fd " << ( fd ? "Y" : "N" ) << std::endl 
       << " path0 " << ( path0 ? path0 : "-" ) << std::endl 
       << " path1 " << ( path1 ? path1 : "-" ) << std::endl 
       << " simtrace0 " << ( simtrace0 ? simtrace0->sstr() : "-" ) << std::endl 
       << " simtrace1 " << ( simtrace1 ? simtrace1->sstr() : "-" ) << std::endl 
       ;

    for(unsigned i=0 ; i < code_count.size() ; i++) ss << " code_count[" << i << "] " << code_count[i] << std::endl ; 
    std::string s = ss.str(); 
    return s ; 
}

inline void SimtraceRerunTest::intersect_again(unsigned idx, bool dump)
{
    const quad4* qq0 = (const quad4*)simtrace0->bytes() ; 
    quad4*  qq1 = (quad4*)simtrace1->bytes() ; 

    const quad4& isect0 = qq0[idx] ; 
    quad4&       isect1 = qq1[idx] ;

    bool valid_isect = q->intersect_again(isect1, isect0); 

    bool valid_isect0 = isect0.q0.f.w > isect0.q1.f.w ;   // dist > tmin
    bool valid_isect1 = isect1.q0.f.w > isect1.q1.f.w ;   // dist > tmin

    unsigned code = ( unsigned(valid_isect0) << 1 ) | unsigned(valid_isect1) ;  
    assert( code < 4 ); 
    code_count[code] += 1 ; 
    code_count[4] += 1 ; 

    if(code == 2 || code == 1)
    {
        std::cout << " idx " << std::setw(7) << idx << " code " << code << std::endl ; 
        std::cout << CSGQuery::Desc( isect0, "isect0", &valid_isect0 ) << std::endl ; 
        std::cout << CSGQuery::Desc( isect1, "isect1", &valid_isect1 ) << std::endl ; 
    }
    assert( valid_isect == valid_isect1 );  
}

inline void SimtraceRerunTest::intersect_again()
{
    unsigned n = simtrace0->shape[0] ; 
    for(unsigned i=0 ; i < n ; i++) 
    {
        bool dump = i % 1000 == 0 ; 
        intersect_again(i, dump); 
    }
}

inline void SimtraceRerunTest::save() const
{
    LOG(info) << " path1 " << path1 ; 
    simtrace1->save(path1); 
}


int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    SimtraceRerunTest t ; 

    t.intersect_again(); 
    t.save(); 

    LOG(info) << "t.desc " << t.desc() ; 

    return 0 ;
}

 
