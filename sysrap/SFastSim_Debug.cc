#include "SFastSim_Debug.hh"
#include "NP.hh"    
#include "SLOG.hh"

const plog::Severity SFastSim_Debug::LEVEL = SLOG::EnvLevel("SFastSim_Debug", "DEBUG" ); 
std::vector<SFastSim_Debug> SFastSim_Debug::record = {} ;

void SFastSim_Debug::Save(const char* dir)
{
    LOG(LEVEL) << " dir " << dir << " num_record " << record.size() ;
    std::cout  
        << "SFastSim_Debug::Save"
        << " dir " << dir 
        << " num_record " << record.size() 
        << std::endl 
        ;
    if( record.size() > 0) NP::Write<double>(dir, NAME, (double*)record.data(), record.size(), NUM_QUAD, 4 );  
    record.clear(); 
}

void SFastSim_Debug::add()
{
    LOG(LEVEL) << "num_record " << record.size() ;
    if(record.size() < LIMIT) record.push_back(*this); 
}

void SFastSim_Debug::fill(double value)
{
    double* ptr = &posx ; 
    for(unsigned i=0 ; i < 4*NUM_QUAD ; i++)  *(ptr + i) = value ; 
}



