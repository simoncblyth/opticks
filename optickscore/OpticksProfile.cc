#include <cstring>
#include <sstream>

#include "SProc.hh"
#include "BTimeStamp.hh"
#include "BTimesTable.hh"
#include "BFile.hh"
#include "BStr.hh"

#include "NGLM.hpp"
#include "NPY.hpp"

#include "OpticksProfile.hh"

#include "PLOG.hh"

OpticksProfile::OpticksProfile(const char* name, bool stamp_out) 
   :
   m_dir(NULL),
   m_name(BStr::concat(NULL,name,".npy")),
   m_columns("Time,DeltaTime,VM,DeltaVM"),
   m_tt(new BTimesTable(m_columns)),
   m_npy(NPY<float>::make(0,1,m_tt->getNumColumns())),

   m_t0(0),
   m_tprev(0),
   m_t(0),

   m_vm0(0),
   m_vmprev(0),
   m_vm(0),

   m_num_stamp(0),
   m_stamp_out(stamp_out)
{
}


void OpticksProfile::setDir(const char* dir)
{
    m_dir = strdup(dir);
}

const char* OpticksProfile::getDir()
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

template <typename T>
void OpticksProfile::stamp(T row, int count)
{
   setT(BTimeStamp::RealTime()) ;
   setVM(SProc::VirtualMemoryUsageMB()) ;
   m_num_stamp += 1 ; 

   float  t   = m_t - m_t0 ;      // time since instanciation
   float dt   = m_t - m_tprev ;   // time since previous stamp

   float vm   = m_vm - m_vm0 ;     // vm since instanciation
   float dvm  = m_vm - m_vmprev ;  // vm since previous stamp

   // the prev start at zero, so first dt and dvm give absolute m_t0 m_vm0 valules

   m_tt->add<T>(row, t, dt, vm, dvm,  count );
   m_npy->add(       t, dt, vm, dvm ); 

   if(m_stamp_out)
   LOG(fatal) << "OpticksProfile::stamp " 
              << m_tt->getLabel() 
              << " (" 
              << t << ","
              << dt << ","
              << vm << ","
              << dvm << ")"
              ; 
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
   assert(dir);
   LOG(error) << brief() ; 
   m_tt->save(dir);
   m_npy->save(dir, m_name);
}
void OpticksProfile::load(const char* dir)
{
   assert(dir);
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


void OpticksProfile::dump(const char* msg, const char* startswith, const char* spacewith, double tcut)
{
    LOG(info) << msg << brief() ; 
  
    m_tt->dump(msg, startswith, spacewith, tcut );
    //m_npy->dump(msg);

    LOG(info) << " npy " << m_npy->getShapeString() << " " << getPath() ; 

}




template OKCORE_API void OpticksProfile::stamp<unsigned>(unsigned , int);
template OKCORE_API void OpticksProfile::stamp<int>(int  , int);
template OKCORE_API void OpticksProfile::stamp<char*>(char*, int);
template OKCORE_API void OpticksProfile::stamp<const char*>(const char*, int);


