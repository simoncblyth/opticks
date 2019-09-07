/*
 * Copyright (c) 2019 Opticks Team. All Rights Reserved.
 *
 * This file is part of Opticks
 * (see https://bitbucket.org/simoncblyth/opticks).
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); 
 * you may not use this file except in compliance with the License.  
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software 
 * distributed under the License is distributed on an "AS IS" BASIS, 
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
 * See the License for the specific language governing permissions and 
 * limitations under the License.
 */


#include <vector>
#include <string>
#include <iostream>
#include <sstream>

#include <boost/lexical_cast.hpp>

#include "PLOG.hh"

#include "BStr.hh"


#ifdef OLD_PARAMETERS
#include "X_BParameters.hh"
#else
#include "NMeta.hpp"
#endif


#include "NOpenMeshEnum.hpp"
#include "NOpenMeshCfg.hpp"


#ifdef OLD_PARAMETERS
NOpenMeshCfg::NOpenMeshCfg(const X_BParameters* meta, const char* treedir) 
#else
NOpenMeshCfg::NOpenMeshCfg(const NMeta* meta, const char* treedir) 
#endif
     : 
     meta(meta),
     treedir(treedir ? strdup(treedir) : NULL),
     level(meta->get<int>("level", "5" )),
     verbosity(meta->get<int>("verbosity", "0" )),
     ctrl(meta->get<int>("ctrl", "0" )),
     poly(meta->get<std::string>("poly", "HY")),
     polycfg(meta->get<std::string>("polycfg", "")),
     combine(NOpenMeshEnum::CombineTypeFromPoly(poly.c_str())),
     epsilon(1e-5f)
{
    dump();
    init();
}


void NOpenMeshCfg::dump(const char* msg)
{
    LOG(info) << msg ; 
    std::cout 
        << " treedir " << ( treedir ? treedir : "-" )
        << " level " << level
        << " verbosity " << verbosity 
        << " ctrl " << ctrl
        << " poly " << poly
        << " polycfg " << polycfg
        << " combine " << combine
        << " epsilon " << epsilon
        << std::endl ; 
}



void NOpenMeshCfg::init()
{
    parse(DEFAULT);
    parse(polycfg.c_str());
}


const char* NOpenMeshCfg::CombineTypeString() const 
{
    return NOpenMeshEnum::CombineType(combine) ;
}



const char* NOpenMeshCfg::CFG_SORTCONTIGUOUS_ = "sortcontiguous" ;
const char* NOpenMeshCfg::CFG_ZERO_ = "zero" ;
const char* NOpenMeshCfg::CFG_CONTIGUOUS_ = "contiguous" ;
const char* NOpenMeshCfg::CFG_PHASED_ = "phased" ;
const char* NOpenMeshCfg::CFG_SPLIT_ = "split" ;
const char* NOpenMeshCfg::CFG_FLIP_ = "flip" ;
const char* NOpenMeshCfg::CFG_NUMFLIP_ = "numflip" ;
const char* NOpenMeshCfg::CFG_MAXFLIP_ = "maxflip" ;
const char* NOpenMeshCfg::CFG_REVERSED_ = "reversed" ;
const char* NOpenMeshCfg::CFG_NUMSUBDIV_ = "numsubdiv" ;
const char* NOpenMeshCfg::CFG_OFFSAVE_ = "offsave" ;

const char* NOpenMeshCfg::DEFAULT = "phased=0,contiguous=0,split=1,flip=1,numflip=0,maxflip=0,reversed=0,sortcontiguous=0,numsubdiv=1,offsave=0" ;

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
    else if(strcmp(k,CFG_NUMSUBDIV_)==0) param = CFG_NUMSUBDIV ;
    else if(strcmp(k,CFG_OFFSAVE_)==0) param = CFG_OFFSAVE ;
    return param ; 
}  

void NOpenMeshCfg::set( NOpenMeshCfgType k, int v )
{
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
        case CFG_NUMSUBDIV      : numsubdiv = v       ; break ; 
        case CFG_OFFSAVE        : offsave = v       ; break ; 
    }
}

std::string NOpenMeshCfg::describe(const char* msg, const char* pfx, const char* kvdelim, const char* delim) const 
{   
    std::stringstream ss ; 
    ss 
        << pfx << msg  << delim 
        << pfx << CFG_CONTIGUOUS_     << kvdelim << contiguous << delim 
        << pfx << CFG_PHASED_         << kvdelim << phased << delim 
        << pfx << CFG_SPLIT_          << kvdelim << split << delim 
        << pfx << CFG_FLIP_           << kvdelim << flip << delim 
        << pfx << CFG_NUMFLIP_        << kvdelim << numflip << delim 
        << pfx << CFG_MAXFLIP_        << kvdelim << maxflip << delim 
        << pfx << CFG_REVERSED_       << kvdelim << reversed << delim 
        << pfx << CFG_SORTCONTIGUOUS_ << kvdelim << sortcontiguous << delim 
        << pfx << CFG_NUMSUBDIV_      << kvdelim << numsubdiv << delim 
        << pfx << CFG_OFFSAVE_        << kvdelim << offsave << delim 
        ;

    return ss.str();
}

std::string NOpenMeshCfg::desc(const char* msg) const 
{
    return describe(msg, " ", ":", "\n" );
}
std::string NOpenMeshCfg::brief(const char* msg) const 
{
    return describe(msg, " ", "=", " " );
}

int NOpenMeshCfg::parse_val(const char* v) const 
{
    return boost::lexical_cast<int>(v);
}  

void NOpenMeshCfg::parse(const char* cfg_)
{
    if(!cfg_) return ; 
    //std::cout << "parsing " << cfg_ << std::endl ; 

    typedef std::pair<std::string,std::string> KV ;
    typedef std::vector<KV>::const_iterator KVI ; 

    std::vector<KV> ekv = BStr::ekv_split(cfg_,',',"=");

    for(KVI it=ekv.begin() ; it!=ekv.end() ; it++)
    {   
        const char* k_ = it->first.c_str() ;   
        const char* v_ = it->second.c_str() ;   

        NOpenMeshCfgType k = parse_key(k_);
        int  v = parse_val(v_);
        set(k,v);
    }
}




