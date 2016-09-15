#include <cstring>
#include <sstream>

#include "SProc.hh"
#include "BTimer.hh"
#include "BStr.hh"

#include "NGLM.hpp"
#include "NPY.hpp"
#include "TimesTable.hpp"

#include "OpticksProfile.hh"

#include "PLOG.hh"

OpticksProfile::OpticksProfile(const char* dir, const char* name) 
   :
   m_dir(strdup(dir)),
   m_name(BStr::concat(NULL,name,".npy")),
   m_columns("Time,DeltaTime,VM,DeltaVM"),
   m_tt(new TimesTable(m_columns)),
   m_npy(NPY<float>::make(0,1,m_tt->getNumColumns())),

   m_t0(0),
   m_tprev(0),
   m_t(0),

   m_vm0(0),
   m_vmprev(0),
   m_vm(0),

   m_num_stamp(0)
{
}


const char* OpticksProfile::getDir()
{
    return m_dir ; 
}
const char* OpticksProfile::getName()
{
    return m_name ; 
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

template <typename T>
void OpticksProfile::stamp(T row, int count)
{
   setT(BTimer::RealTime()) ;
   setVM(SProc::VirtualMemoryUsageMB()) ;
   m_num_stamp += 1 ; 

   float  t   = m_t - m_t0 ;      // time since instanciation
   float dt   = m_t - m_tprev ;   // time since previous stamp

   float vm   = m_vm - m_vm0 ;     // vm since instanciation
   float dvm  = m_vm - m_vmprev ;  // vm since previous stamp

   m_tt->add<T>(row, t, dt, vm, dvm,  count );
   m_npy->add(       t, dt, vm, dvm ); 
}



void OpticksProfile::save()
{
   save(m_dir); 
}
void OpticksProfile::load()
{
   load(m_dir); 
}

void OpticksProfile::save(const char* dir)
{
   m_tt->save(dir);
   m_npy->save(dir, m_name);
}
void OpticksProfile::load(const char* dir)
{
   m_tt->load(dir);
   m_npy = NPY<float>::load(dir, m_name);
}

std::string OpticksProfile::brief()
{
   std::stringstream ss ;
   ss
       << " dir " << m_dir 
       << " name " << m_name
       << " num_stamp " << m_num_stamp 
       ; 
   
   return ss.str();
}


void OpticksProfile::dump(const char* msg)
{
    LOG(info) << msg << brief() ; 
  
    m_tt->dump(msg);
    m_npy->dump(msg);
}




template OKCORE_API void OpticksProfile::stamp<unsigned>(unsigned , int);
template OKCORE_API void OpticksProfile::stamp<int>(int  , int);
template OKCORE_API void OpticksProfile::stamp<char*>(char*, int);
template OKCORE_API void OpticksProfile::stamp<const char*>(const char*, int);


