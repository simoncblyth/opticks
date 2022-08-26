/**
CSGFoundrySimtraceRerunTest.cc
================================

Roughly based on CSGQueryTest.cc but updated for simtrace array reruns

TODO: 

* shoot ray, trying to duplicate the simtrace item

**/


#include "SSys.hh"
#include "SSim.hh"
#include "SPath.hh"
#include "NP.hh"
#include "OPTICKS_LOG.hh"

#include "CSGFoundry.h"
//#include "CSGQuery.h"
//#include "CSGDraw.h"


struct CSGFoundrySimtraceRerunTest
{ 
    SSim* sim ; 
    const CSGFoundry* fd ; 
    const char* path ; 
    NP* simtrace ; 

    CSGFoundrySimtraceRerunTest(); 
    std::string desc() const ; 
};

inline CSGFoundrySimtraceRerunTest::CSGFoundrySimtraceRerunTest()
    :
    sim(SSim::Create()),
    fd(CSGFoundry::Load()),
    path(SPath::Resolve("$T_FOLD/simtrace.npy", NOOP)),
    simtrace(NP::Load(path))
{
}
 
inline std::string CSGFoundrySimtraceRerunTest::desc() const
{
    std::stringstream ss ; 
    ss << "CSGFoundrySimtraceRerunTest::desc" << std::endl 
       << " fd " << ( fd ? "Y" : "N" ) << std::endl 
       << " path " << ( path ? path : "-" ) << std::endl 
       << " simtrace " << ( simtrace ? simtrace->sstr() : "-" ) << std::endl 
       ;
    std::string s = ss.str(); 
    return s ; 
}

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    CSGFoundrySimtraceRerunTest t ; 
    LOG(info) << "t.desc " << t.desc() ; 

    return 0 ;
}

 
