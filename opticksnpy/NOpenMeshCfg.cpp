
#include <vector>
#include <string>
#include <iostream>
#include <sstream>

#include <boost/lexical_cast.hpp>

#include "BStr.hh"

#include "NOpenMeshCfg.hpp"


const char* NOpenMeshCfg::CFG_SORTCONTIGUOUS_ = "sortcontiguous" ;
const char* NOpenMeshCfg::CFG_ZERO_ = "zero" ;
const char* NOpenMeshCfg::CFG_CONTIGUOUS_ = "contiguous" ;
const char* NOpenMeshCfg::CFG_PHASED_ = "phased" ;
const char* NOpenMeshCfg::CFG_SPLIT_ = "split" ;
const char* NOpenMeshCfg::CFG_FLIP_ = "flip" ;
const char* NOpenMeshCfg::CFG_NUMFLIP_ = "numflip" ;
const char* NOpenMeshCfg::CFG_MAXFLIP_ = "maxflip" ;
const char* NOpenMeshCfg::CFG_REVERSED_ = "reversed" ;

const char* NOpenMeshCfg::DEFAULT = "phased=0,contiguous=0,split=1,flip=1,numflip=0,maxflip=0,reversed=0,sortcontiguous=0" ;

NOpenMeshCfgType  NOpenMeshCfg::parse_key(const char* k) const 
{
    NOpenMeshCfgType param = CFG_ZERO ; 

    if(     strcmp(k,CFG_CONTIGUOUS_)==0) param = CFG_CONTIGUOUS ;
    else if(strcmp(k,CFG_PHASED_)==0)     param = CFG_PHASED ;
    else if(strcmp(k,CFG_SPLIT_)==0)      param = CFG_SPLIT ;
    else if(strcmp(k,CFG_FLIP_)==0)       param = CFG_FLIP ;
    else if(strcmp(k,CFG_NUMFLIP_)==0)    param = CFG_NUMFLIP ;
    else if(strcmp(k,CFG_MAXFLIP_)==0)    param = CFG_MAXFLIP ;
    else if(strcmp(k,CFG_REVERSED_)==0)   param = CFG_REVERSED ;
    else if(strcmp(k,CFG_SORTCONTIGUOUS_)==0) param = CFG_SORTCONTIGUOUS ;
    return param ; 
}  

int NOpenMeshCfg::parse_val(const char* v) const 
{
    return boost::lexical_cast<int>(v);
}  


NOpenMeshCfg::NOpenMeshCfg(const char* cfg_) 
     : 
     cfg(cfg_ ? strdup(cfg_) : NULL )
{
    init();
}

void NOpenMeshCfg::init()
{
    parse(DEFAULT);
    parse(cfg);
}


void NOpenMeshCfg::parse(const char* cfg_)
{
    if(!cfg_) return ; 
    std::cout << "parsing " << cfg_ << std::endl ; 

    typedef std::pair<std::string,std::string> KV ;
    typedef std::vector<KV>::const_iterator KVI ; 

    std::vector<KV> ekv = BStr::ekv_split(cfg_,',',"=");

    for(KVI it=ekv.begin() ; it!=ekv.end() ; it++)
    {   
        const char* k_ = it->first.c_str() ;   
        const char* v_ = it->second.c_str() ;   

        NOpenMeshCfgType k = parse_key(k_);
        int  v = parse_val(v_);

        switch(k)
        {
            case CFG_ZERO           : assert(0)      ; break ; 
            case CFG_CONTIGUOUS     : contiguous = v ; break ; 
            case CFG_PHASED         : phased = v     ; break ; 
            case CFG_SPLIT          : split = v      ; break ; 
            case CFG_FLIP           : flip = v       ; break ; 
            case CFG_NUMFLIP        : numflip = v       ; break ; 
            case CFG_MAXFLIP        : maxflip = v       ; break ; 
            case CFG_REVERSED       : reversed = v       ; break ; 
            case CFG_SORTCONTIGUOUS : sortcontiguous = v ; break ; 
        }
    }
}

std::string NOpenMeshCfg::desc(const char* msg) const 
{   
    std::stringstream ss ; 
    ss 
        << msg  << std::endl 
        << " " << CFG_CONTIGUOUS_ << ":" << contiguous << std::endl 
        << " " << CFG_PHASED_     << ":" << phased << std::endl 
        << " " << CFG_SPLIT_      << ":" << split << std::endl 
        << " " << CFG_FLIP_       << ":" << flip << std::endl 
        << " " << CFG_NUMFLIP_    << ":" << numflip << std::endl 
        << " " << CFG_MAXFLIP_    << ":" << maxflip << std::endl 
        << " " << CFG_REVERSED_   << ":" << reversed << std::endl 
        << " " << CFG_SORTCONTIGUOUS_ << ":" << sortcontiguous << std::endl 
        ;

    return ss.str();
}

