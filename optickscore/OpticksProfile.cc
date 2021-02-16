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

#include <cstring>
#include <csignal>
#include <sstream>

#include "SProc.hh"
#include "BTimeStamp.hh"
#include "BTimesTable.hh"
#include "BFile.hh"
#include "BStr.hh"

#include "NPY.hpp"

#include "OpticksProfile.hh"

#include "PLOG.hh"

const plog::Severity OpticksProfile::LEVEL = PLOG::EnvLevel("OpticksProfile", "DEBUG") ; 

const char* OpticksProfile::NAME = "OpticksProfile" ; 

/**

TODO 

* get rid of m_tt is almost been fully replaced by the NPY arrays 

**/

OpticksProfile::OpticksProfile() 
    :
    m_stamp(false),
    m_dir(NULL),
    m_name(BStr::concat(NULL,NAME,".npy")),
    m_lname(BStr::concat(NULL,NAME,"Labels.npy")),
    m_aname(BStr::concat(NULL,NAME,"Acc.npy")),
    m_laname(BStr::concat(NULL,NAME,"AccLabels.npy")),

    m_qname(BStr::concat(NULL,NAME,"Lis.npy")),
    m_lqname(BStr::concat(NULL,NAME,"LisLabels.npy")),


    m_columns("Time,DeltaTime,VM,DeltaVM"),
    m_tt(new BTimesTable(m_columns)),
    m_npy(NPY<float>::make(0,m_tt->getNumColumns())),
    m_lpy(NPY<char>::make(0,64)),
    m_apy(NPY<float>::make(0,4)),
    m_lapy(NPY<char>::make(0,64)),

    m_qpy(NPY<double>::make(0,4)),
    m_lqpy(NPY<char>::make(0,64)),

    m_t0(0),
    m_tprev(0),
    m_t(0),

    m_vm0(0),
    m_vmprev(0),
    m_vm(0),

    m_num_stamp(0),
    m_last_stamp(0.f, 0.f, 0.f, 0.f)
{
}


void OpticksProfile::setStamp(bool stamp)
{
    m_stamp = stamp ; 
}


std::vector<std::string>&  OpticksProfile::getLines()
{
    return m_tt->getLines(); 
}


/**
OpticksProfile::setMeta
------------------------

NOT CURRENTLY USED.  

Considered adding numPhotons etc.. but there
is so much more like number and types of GPU etc.., 
versions of Opticks and Geant4 etc..
that it makes no sense to duplicate a subset of the 
full parameters here.

**/

template<typename T>
void OpticksProfile::setMeta(const char* key, T value )
{
    m_npy->setMeta(key, value); 
}

template<typename T>
T OpticksProfile::getMeta(const char* key, const char* fallback) const
{
    return m_npy->getMeta<T>(key, fallback); 
}



void OpticksProfile::setDir(const char* dir)
{
    LOG(LEVEL) << "dir " << m_dir ;  
    m_dir = strdup(dir);
    //std::raise(SIGINT); 
}

const char* OpticksProfile::getDir() const
{
    return m_dir ; 
}
const char* OpticksProfile::getName()
{
    return m_name ; 
}
std::string OpticksProfile::getPath()
{
    return BFile::FormPath(m_dir, m_name);
}


void OpticksProfile::setT(float t)
{
   if(m_num_stamp == 0) m_t0 = t ;
   m_tprev = m_t ;  
   m_t  = t ;   
}
void OpticksProfile::setVM(float vm)
{
   if(m_num_stamp == 0) m_vm0 = vm ;
   m_vmprev = m_vm ;  
   m_vm = vm ; 
}


/**
OpticksProfile::stamp
-----------------------

Called from Opticks::profile canonically with 
const char* labels from OK_PROFILE invokations.

**/

template <typename T>
void OpticksProfile::stampOld(T label, int count)
{
   setT(BTimeStamp::RealTime()) ;
   setVM(SProc::VirtualMemoryUsageMB()) ;
   m_num_stamp += 1 ; 

   float  t   = m_t - m_t0 ;      // time since instanciation
   float dt   = m_t - m_tprev ;   // time since previous stamp

   float vm   = m_vm - m_vm0 ;     // vm since instanciation
   float dvm  = m_vm - m_vmprev ;  // vm since previous stamp

   // the prev start at zero, so first dt and dvm give absolute m_t0 m_vm0 valules

   m_tt->add<T>(label, t, dt, vm, dvm,  count );
   m_npy->add(       t, dt, vm, dvm ); 
   //m_lpy->addString( label ) ; 

   LOG(LEVEL)
       << m_tt->getLabel() 
       << " (" 
       << t << ","
       << dt << ","
       << vm << ","
       << dvm << ")"
       ; 
}

/**
OpticksProfile::stamp
------------------------

Hmm the NPY::add will have to do memcpy sometimes, 
maybe cause of 0.0039s glitches

TODO: minimize whats done in here 

**/

void OpticksProfile::stamp(const char* label, int count)
{
   setT(BTimeStamp::RealTime2()) ;
   setVM(SProc::VirtualMemoryUsageMB()) ;
   m_num_stamp += 1 ; 

   float  t   = m_t - m_t0 ;      // time since instanciation
   float dt   = m_t - m_tprev ;   // time since previous stamp

   float vm   = m_vm - m_vm0 ;     // vm since instanciation
   float dvm  = m_vm - m_vmprev ;  // vm since previous stamp

   m_last_stamp.x = t ;
   m_last_stamp.y = dt ;
   m_last_stamp.z = vm ;
   m_last_stamp.w = dvm ;

   // the prev start at zero, so first dt and dvm give absolute m_t0 m_vm0 valules

   m_tt->add<const char*>(label, t, dt, vm, dvm,  count );
   m_npy->add(       t, dt, vm, dvm ); 
   m_lpy->addString( label ) ; 

   LOG(LEVEL)
       << m_tt->getLabel() 
       << " (" 
       << t << ","
       << dt << ","
       << vm << ","
       << dvm << ")"
       ; 
}

glm::vec4 OpticksProfile::Stamp() // static
{
    glm::vec4 stamp ; 
    stamp.x = BTimeStamp::RealTime2() ; 
    stamp.y = SProc::VirtualMemoryUsageMB() ; 
    stamp.z = 0.f ; 
    stamp.w = 0.f ; 
    return stamp ; 
}



const glm::vec4& OpticksProfile::getLastStamp() const 
{
    return m_last_stamp ; 
}



unsigned OpticksProfile::lisAdd(const char* label)
{
    unsigned idx = m_lis.size(); 

    m_lis_labels.push_back(label); 

    OpticksLis ls ; 
    m_lis.push_back(ls);  

    return idx ; 
}


void OpticksProfile::lisAppend(unsigned idx, double t )
{
    OpticksLis& lis = m_lis[idx] ; 

    lis.tt.push_back(t) ; 
}







unsigned OpticksProfile::accumulateAdd(const char* label)
{
    unsigned idx = m_acc.size(); 

    m_acc_labels.push_back(label); 

    OpticksAcc acc ; 
    OpticksAcc::Init(acc);  
    m_acc.push_back(acc);  

    return idx ; 
}

void OpticksProfile::accumulateStart(unsigned idx)
{
    OpticksAcc& acc = m_acc[idx] ; 
    acc.t0 = BTimeStamp::RealTime() ; 
    acc.v0 = SProc::VirtualMemoryUsageKB() ; 
}

void OpticksProfile::accumulateStop(unsigned idx)
{
    OpticksAcc& acc = m_acc[idx] ; 

    float dt = BTimeStamp::RealTime() - acc.t0 ; 
    float dv = SProc::VirtualMemoryUsageKB() - acc.v0 ; 

    acc.n += 1 ; 
    acc.t += dt ; 
    acc.v += dv ; 

    if(acc.n % 1000 == 0 )
    {
        LOG(LEVEL) << accumulateDesc(idx) ; 
    }
}





void OpticksProfile::accumulateSet(unsigned idx, float dt )
{
    OpticksAcc& acc = m_acc[idx] ; 

    acc.n += 1 ; 
    acc.t += dt ; 
    acc.v += 0.f ; 
}




std::string OpticksProfile::accumulateDesc(unsigned idx) const 
{
    const OpticksAcc& acc = m_acc[idx] ; 
    std::stringstream ss ; 
    ss 
       << "Acc "    
       << std::setw(50) << m_acc_labels[idx]
       << " n " << std::setw(9) << acc.n 
       << " t " << std::setw(9) << std::fixed << std::setprecision(4) << acc.t 
       << " v " << std::setw(9) << std::fixed << std::setprecision(4) << acc.v
       ; 
    return ss.str(); 
}


void OpticksProfile::accumulateDump(const char* msg) const 
{
    unsigned nacc = m_acc.size() ; 
    LOG(info) << msg << " nacc " << nacc ; 
    for(unsigned i=0 ; i < nacc ; i++) 
        std::cout 
            << std::setw(4) << std::setfill('0') << i 
            << " : " 
            << accumulateDesc(i) 
            << std::endl
             ; 
}


bool OpticksProfile::isAccExported() const 
{
    unsigned n_apy = m_apy->getNumItems() ;
    unsigned n_lapy = m_lapy->getNumItems() ;
    bool match = n_apy == n_lapy ; 
    if(!match)
       LOG(fatal) 
            << "MISMATCH"
            << " apy " << m_apy->getShapeString()
            << " lapy " << m_lapy->getShapeString()
            ;

    assert( match ); 
    return n_apy > 0 ; 
}

void OpticksProfile::accumulateExport() 
{
    if(isAccExported()) return ; 

    unsigned nacc = m_acc.size() ; 
    LOG(LEVEL) << " nacc " << nacc ; 
    assert( m_acc_labels.size() == nacc ); 


    for(unsigned idx=0 ; idx < nacc ; idx++) 
    {
        const OpticksAcc& acc = m_acc[idx] ; 
        m_apy->add( float(acc.n), acc.t, acc.v,  0.f ); 

        const std::string& label = m_acc_labels[idx]; 
        m_lapy->addString(label.c_str()); 
    }


    unsigned nlis = m_lis.size() ; 
    LOG(LEVEL) << " nlis " << nlis ; 
    assert( m_lis_labels.size() == nlis ); 

    // TODO: handle more than one list using 
    //       2d NPY with zero padding to the longest ?
    //       then can tend to use for event level things
    //       so will line up anyhow as same numbers of events 
    //
    //       so labels will be of the rows   

    assert( nlis < 2 ); 
    if( nlis == 1)
    {
        unsigned idx = 0 ; 
        const OpticksLis& lis = m_lis[idx] ; 
        const std::vector<double>& tt = lis.tt ;   
        m_qpy = NPY<double>::make_from_vec(tt) ; 
        const std::string& label = m_lis_labels[idx]; 
        m_lqpy->addString(label.c_str()); 
    }

}



void OpticksProfile::save()
{
    LOG(LEVEL) << "dir " << m_dir ;  
    save(m_dir); 
}
void OpticksProfile::load()
{
    load(m_dir); 
}



void OpticksProfile::save(const char* dir)
{
   assert(dir);
   LOG(LEVEL) << brief() ; 

   accumulateExport(); 

   m_tt->save(dir);
   m_npy->save(dir, m_name);
   m_lpy->save(dir, m_lname);
   m_apy->save(dir, m_aname);
   m_lapy->save(dir, m_laname);

   m_qpy->save(dir, m_qname);
   m_lqpy->save(dir, m_lqname);


}



OpticksProfile* OpticksProfile::Load( const char* dir )
{
    OpticksProfile* profile = new OpticksProfile();
    profile->setDir(dir); 
    profile->load(); 
    return profile ;  
}


void OpticksProfile::load(const char* dir)
{
   assert(dir);
   //m_tt->load(dir);
   m_npy = NPY<float>::load(dir, m_name);
   m_lpy = NPY<char>::load(dir, m_lname);
}

std::string OpticksProfile::brief()
{
   std::stringstream ss ;
   ss
       << " dir " << ( m_dir ? m_dir : "-" ) 
       << " name " << m_name
       << " num_stamp " << m_num_stamp 
       ; 
   
   return ss.str();
}


void OpticksProfile::dump(const char* msg, const char* startswith, const char* spacewith, double tcut)
{
    LOG(info) << msg << brief() ; 
 
    if(m_tt) 
    m_tt->dump(msg, startswith, spacewith, tcut );
    //m_npy->dump(msg);

    LOG(info) << " npy " << m_npy->getShapeString() << " " << getPath() ; 

    accumulateDump(msg); 
}


void OpticksProfile::Report(const NPY<float>* a, float profile_leak_mb)  // static
{
    unsigned num_stamp = a->getNumItems(); 

    float t0 = a->getValue( 0, 0, 0) ; 
    float t1 = a->getValue(-1, 0, 0) ; 
    float dt = t1 - t0 ; 
    float dt_per_stamp = dt/float(num_stamp-1);   // subtract 1 because profile stamps only at one point  

    float v0 = a->getValue( 0, 1, 0) ; 
    float v1 = a->getValue(-1, 1, 0) ; 
    float dv = v1 - v0 ; 
    float dv_per_stamp = dv/float(num_stamp-1);    // subtract 1 because profile stamps only at one point  

    LOG(info) 
        << " num_stamp " << num_stamp
        << " profile_leak_mb " << profile_leak_mb
        << "    "
        << " t0 " << t0   
        << " t1 " << t1 
        << " dt " << dt 
        << " dt/(num_stamp-1) " << dt_per_stamp
        << "    "
        << " v0 (MB) " << v0 
        << " v1 (MB) " << v1 
        << " dv " << dv 
        << " dv/(num_stamp-1) " << dv_per_stamp
        ;
}




template OKCORE_API void OpticksProfile::stampOld<unsigned>(unsigned , int);
template OKCORE_API void OpticksProfile::stampOld<int>(int  , int);
template OKCORE_API void OpticksProfile::stampOld<char*>(char*, int);
template OKCORE_API void OpticksProfile::stampOld<const char*>(const char*, int);


