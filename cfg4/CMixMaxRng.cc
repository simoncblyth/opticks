#include <fstream>

#include "Randomize.hh"
#include "CMixMaxRng.hh"
#include "SBacktrace.hh"
#include "PLOG.hh"


CMixMaxRng::CMixMaxRng()
    :
    count(0), 
    out(NULL)
{
    CLHEP::HepRandom::setTheEngine( this );  

    //out = new std::ostream(std::cout.rdbuf()) ;
    out = new std::ofstream("/tmp/simstream.txt") ;

}


/**
CMixMaxRng::flat
-------------------

Instrumented shim for flat,  

Finding the CallSite in the backtrace with "::flat" 
matches either ::flat() or ::flatArray(.. 
so get the line following those.


**/

double CMixMaxRng::flat()
{
    double v = CLHEP::MixMaxRng::flat(); 

    if(count == 0)
        SBacktrace::Dump();

    const char* caller = SBacktrace::CallSite( "::flat" ) ; 

    (*out) 
        << std::setw(6) << count 
        << " : " 
        << std::setw(10) << std::fixed << v 
        << " : "
        << caller
        << std::endl 
        ;

    count += 1 ; 
    return v ;   
}


void CMixMaxRng::preTrack() 
{
    LOG(info) << "." ; 
} 
void CMixMaxRng::postTrack()
{
    LOG(info) << "." ; 
} 
void CMixMaxRng::postStep() 
{
    LOG(info) << "." ; 
}
void CMixMaxRng::postpropagate()
{
    LOG(info) << "." ; 
}
double CMixMaxRng::flat_instrumented(const char* file, int line)
{
    LOG(info) << file << ":" << line ; 
    return flat(); 
}



