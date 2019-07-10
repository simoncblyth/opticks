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

const plog::Severity OpticksProfile::LEVEL = PLOG::EnvLevel("OpticksProfile", "DEBUG") ; 

const char* OpticksProfile::NAME = "OpticksProfile" ; 

OpticksProfile::OpticksProfile() 
    :
    m_stamp(false),
    m_dir(NULL),
    m_name(BStr::concat(NULL,NAME,".npy")),
    m_lname(BStr::concat(NULL,NAME,"Labels.npy")),
    m_columns("Time,DeltaTime,VM,DeltaVM"),
    m_tt(new BTimesTable(m_columns)),
    m_npy(NPY<float>::make(0,1,m_tt->getNumColumns())),
    m_lpy(NPY<char>::make(0,1,64)),

    m_t0(0),
    m_tprev(0),
    m_t(0),

    m_vm0(0),
    m_vmprev(0),
    m_vm(0),

    m_num_stamp(0)
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


void OpticksProfile::stamp(const char* label, int count)
{
   setT(BTimeStamp::RealTime()) ;
   setVM(SProc::VirtualMemoryUsageMB()) ;
   m_num_stamp += 1 ; 

   float  t   = m_t - m_t0 ;      // time since instanciation
   float dt   = m_t - m_tprev ;   // time since previous stamp

   float vm   = m_vm - m_vm0 ;     // vm since instanciation
   float dvm  = m_vm - m_vmprev ;  // vm since previous stamp

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

std::string OpticksProfile::accumulateDesc(unsigned idx)
{
    OpticksAcc& acc = m_acc[idx] ; 
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
   LOG(LEVEL) << brief() ; 
   m_tt->save(dir);
   m_npy->save(dir, m_name);
   m_lpy->save(dir, m_lname);
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
   m_tt->load(dir);
   m_npy = NPY<float>::load(dir, m_name);
   m_lpy = NPY<char>::load(dir, m_lname);
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




template OKCORE_API void OpticksProfile::stampOld<unsigned>(unsigned , int);
template OKCORE_API void OpticksProfile::stampOld<int>(int  , int);
template OKCORE_API void OpticksProfile::stampOld<char*>(char*, int);
template OKCORE_API void OpticksProfile::stampOld<const char*>(const char*, int);


