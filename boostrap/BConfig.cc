
#include "BStr.hh"
#include "BConfig.hh"
#include "PLOG.hh"

BConfig::BConfig(const char* cfg_) 
    :   
    cfg(cfg_ ? strdup(cfg_) : NULL)
{
}


//template <typename T> void BConfig::add<T>(const char* k, T* ptr)

void BConfig::addInt(const char* k, int* ptr)
{
    eki.push_back(KI(k, ptr)); 
}

void BConfig::addFloat(const char* k, float* ptr)
{
    ekf.push_back(KF(k, ptr)); 
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

        for(unsigned j=0 ; j < ekf.size() ; j++)
        {   
            KF kf = ekf[j] ; 
            if(strcmp(kf.first.c_str(),kv.first.c_str()) == 0) *kf.second = BStr::atof(kv.second.c_str()) ; 
        }   

    }   
    //dump("BConfig::parse");
}

void BConfig::dump(const char* msg) const 
{
    LOG(info) << msg  ; 

    dump_eki();
    dump_ekf();
    dump_ekv();

}

void BConfig::dump_eki() const 
{
    LOG(info) << " eki " << eki.size() ; 
    for(unsigned j=0 ; j < eki.size() ; j++)
    {   
        KI ki = eki[j] ; 
        std::cout << std::setw(40) << ki.first << " : " << *ki.second << std::endl ;
    }   
}

void BConfig::dump_ekf() const 
{
    LOG(info) << " ekf " << ekf.size() ; 
    for(unsigned i=0 ; i < ekf.size() ; i++)
    {   
        KF kf = ekf[i] ; 
        std::cout << std::setw(30) << kf.first << " : " << std::setw(20) << *kf.second << std::endl ; 
    }
}

void BConfig::dump_ekv() const 
{
    LOG(info) << " ekv " << ekv.size() ; 
    for(unsigned i=0 ; i < ekv.size() ; i++)
    {   
        KV kv = ekv[i] ; 
        std::cout << std::setw(30) << kv.first << " : " << std::setw(20) << kv.second << std::endl ; 
    }
}



