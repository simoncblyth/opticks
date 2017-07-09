
#include "BStr.hh"
#include "BConfig.hh"
#include "PLOG.hh"

BConfig::BConfig(const char* cfg) 
    :   
    cfg(cfg ? strdup(cfg) : NULL)
{
}


//template <typename T> void BConfig::add<T>(const char* k, T* ptr)

void BConfig::addInt(const char* k, int* ptr)
{
    eki.push_back(KI(k, ptr)); 
}


void BConfig::parse()
{
    if(!cfg) return ; 
    BStr::ekv_split(ekv, cfg, ',', "=" );
    for(unsigned i=0 ; i < ekv.size() ; i++)
    {   
        KV kv = ekv[i] ; 
        for(unsigned j=0 ; j < eki.size() ; j++)
        {   
            KI ki = eki[j] ; 
            if(strcmp(ki.first.c_str(),kv.first.c_str()) == 0) *ki.second = BStr::atoi(kv.second.c_str()) ; 
        }   
    }   
    dump_ekv();
}

void BConfig::dump(const char* msg) const 
{
    LOG(info) << msg << " eki " << eki.size() ; 
    for(unsigned j=0 ; j < eki.size() ; j++)
    {   
        KI ki = eki[j] ; 
        std::cout << std::setw(40) << ki.first << " : " << *ki.second << std::endl ;
    }   
}

void BConfig::dump_ekv() const 
{
    for(unsigned i=0 ; i < ekv.size() ; i++)
    {   
        KV kv = ekv[i] ; 
        std::cout 
             << std::setw(30) << kv.first
             << " : "
             << std::setw(20) << kv.second
             << std::endl
             ;
    }
}

