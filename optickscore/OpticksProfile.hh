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

#pragma once

#include <vector>
#include <string>
class BTimesTable ; 
template <typename T> class NPY ; 

#include "NGLM.hpp"
#include "OKCORE_API_EXPORT.hh"
#include "OKCORE_HEAD.hh"
#include "plog/Severity.h"

/**

OpticksProfile
===============

Canonical m_profile instance is resident of Opticks and
is instanciated with it.



Recording time and virtual memory as various points during Opticks running.

Ideas to improve stopwatch 

* https://codereview.stackexchange.com/questions/196245/extremely-simple-timer-class-in-c



**/

struct OKCORE_API OpticksAcc
{
    unsigned n ; 
    float    t ; 
    float    v ; 
    float    t0 ; 
    float    v0 ; 

    static void Init(OpticksAcc& acc )
    { 
        acc.n = 0 ; 
        acc.t = 0.f ; 
        acc.v = 0.f ; 
        acc.t0 = 0.f ; 
        acc.v0 = 0.f ;
    }  
};


struct OKCORE_API OpticksLis
{
    std::vector<double> tt ; 
};




class OKCORE_API OpticksProfile 
{
    private:
       static const plog::Severity LEVEL ; 
       static const char* NAME ; 
    public:
       static OpticksProfile* Load( const char* dir); 
    public:
       OpticksProfile();
       template <typename T> void stampOld(T row, int count);
       void stamp(const char* label, int count);

       unsigned lisAdd(const char* label); 
       void lisAppend(unsigned idx, double t ); 

       unsigned accumulateAdd(const char* label); 
       void     accumulateStart(unsigned idx); 
       void     accumulateStop(unsigned idx); 
       std::string accumulateDesc(unsigned idx) const ;

       void     accumulateSet(unsigned idx, float dt); 


       void     accumulateDump(const char* msg) const ;
       void     accumulateExport()  ;
       bool     isAccExported() const ;

       void setStamp(bool stamp); 

       std::vector<std::string>&  getLines(); 
       void save();
       void load();
       void dump(const char* msg="OpticksProfile::dump", const char* startswith=NULL, const char* spacewith=NULL, double tcut=0.0);

       template<typename T>
       void setMeta(const char* key, T value ); 

       template<typename T>
       T getMeta(const char* key, const char* fallback) const ; 


       void setDir(const char* dir);
       const char* getDir() const ;
       const char* getName();
       std::string getPath();

       std::string brief();

       const glm::vec4& getLastStamp() const ; 

    public:
       void save(const char* dir);
    private:
       void setT(float t);
       void setVM(float vm);
       void load(const char* dir);
    private:
       bool        m_stamp ;  
       const char* m_dir ; 
       const char* m_name ; 
       const char* m_lname ; 
       const char* m_aname ; 
       const char* m_laname ; 

       const char* m_qname ; 
       const char* m_lqname ; 



       const char* m_columns ; 
       BTimesTable* m_tt ; 

       NPY<float>* m_npy ;
       NPY<char>*  m_lpy ;
       NPY<float>* m_apy ;
       NPY<char>*  m_lapy ;
       NPY<double>* m_qpy ;
       NPY<char>*  m_lqpy ;
 
 
       float       m_t0 ; 
       float       m_tprev ; 
       float       m_t ; 

       float       m_vm0 ; 
       float       m_vmprev ; 
       float       m_vm ; 

       unsigned    m_num_stamp ; 

       std::vector<OpticksAcc>   m_acc ; 
       std::vector<std::string>  m_acc_labels  ; 

       std::vector<OpticksLis>   m_lis ; 
       std::vector<std::string>  m_lis_labels  ; 

       glm::vec4  m_last_stamp ; 


};

#include "OKCORE_TAIL.hh"



