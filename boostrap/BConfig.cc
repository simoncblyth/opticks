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


#include "BStr.hh"
#include "BConfig.hh"
#include "PLOG.hh"

const char* BConfig::DEFAULT_KVDELIM = "=" ; 


BConfig::BConfig(const char* cfg_, char edelim_, const char* kvdelim_) 
    :   
    cfg(cfg_ ? strdup(cfg_) : NULL),
    edelim( edelim_ ? edelim_ : ',' ),
    kvdelim( kvdelim_ ? strdup(kvdelim_) : DEFAULT_KVDELIM )
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
void BConfig::addString(const char* k, std::string* ptr)
{
    eks.push_back(KS(k, ptr)); 
}



void BConfig::parse()
{
    if(!cfg) return ; 

    int err = BStr::ekv_split(ekv, cfg, edelim,  kvdelim );
    bool valid = err == 0 ; 

    if(!valid)
    {
        LOG(fatal) << "BConfig::parse err from BStr::ekv_split "
                   << " cfg [" << cfg  << "]" 
                   ;
    }
    assert( valid );

    for(unsigned i=0 ; i < ekv.size() ; i++)
    {   
        KV kv = ekv[i] ; 

        const char* k = kv.first.c_str(); 
        const char* v = kv.second.c_str(); 

        int found(0); 

        for(unsigned j=0 ; j < eki.size() ; j++)
        {   
            KI ki = eki[j] ; 
            if(strcmp(ki.first.c_str(),k) == 0)
            { 
                found += 1 ; 
                *ki.second = BStr::atoi(v) ; 
            }
        }   

        for(unsigned j=0 ; j < ekf.size() ; j++)
        {   
            KF kf = ekf[j] ; 
            if(strcmp(kf.first.c_str(),k) == 0) 
            { 
                found += 1 ; 
                *kf.second = BStr::atof(v) ; 
            }
        }   

        for(unsigned j=0 ; j < eks.size() ; j++)
        {   
            KS ks = eks[j] ; 
            if(strcmp(ks.first.c_str(),k) == 0) 
            {
                found += 1 ; 
                *ks.second = v ; 
            }
        }   


        if(found != 1)
        {
            LOG(fatal) 
               << " UNKNOWN/DUPLICATE KEY " << k << " : " << v 
               << " found " << found 
               << " in config " << cfg
               ;
        } 
        assert( found == 1);



    }   
    //dump("BConfig::parse");
}

void BConfig::dump(const char* msg) const 
{
    LOG(info) << msg  ; 

    dump_ekv();

    dump_eki();
    dump_ekf();
    dump_eks();

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

void BConfig::dump_eks() const 
{
    LOG(info) << " eks " << eks.size() ; 
    for(unsigned i=0 ; i < eks.size() ; i++)
    {   
        KS ks = eks[i] ; 
        std::cout << std::setw(30) << ks.first << " : " << std::setw(20) << *ks.second << std::endl ; 
    }
}

std::string BConfig::desc() const 
{
    std::stringstream ss ; 
    ss
       << " BConfig.initial " << ( cfg ? cfg : "-" )
       << " ekv " << ekv.size()
       << " eki " << eki.size()
       << " ekf " << ekf.size()
       << " eks " << eks.size()
       ;
    return ss.str();
}





