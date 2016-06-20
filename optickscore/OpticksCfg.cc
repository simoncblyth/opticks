
#include <string>
#include <vector>

#include "BLog.hh"

#include "OpticksCfg.hh"

#include "Opticks.hh"

template <class Listener>
void OpticksCfg<Listener>::dump(const char* msg)
{
    LOG(info) << msg ;
    
    std::vector<std::string> names ; 
    names.push_back("compute");
    names.push_back("save");
    names.push_back("load");
    names.push_back("test");

    names.push_back("g4gun");

    for(unsigned int i=0 ; i < names.size() ; i++)
    {
        LOG(info) << std::setw(10) << names[i] << " " <<  hasOpt(names[i].c_str()) ;
    }
}



template class OpticksCfg<Opticks> ;
