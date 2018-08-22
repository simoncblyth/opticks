#include "Randomize.hh"
#include "CMixMaxRng.hh"
#include "SBacktrace.hh"
#include "PLOG.hh"


CMixMaxRng::CMixMaxRng()
    :
    count(0)
{
    CLHEP::HepRandom::setTheEngine( this );  
}



double CMixMaxRng::flat()
{
    double v = CLHEP::MixMaxRng::flat(); 
    LOG(info) 
        << std::setw(6) << count 
        << " : " 
        << v 
        ;

    if(count == 0)
    {
        SBacktrace::Dump();
        assert(0); 
    }

 
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



