#include "SProc.hh"

#include "NGLM.hpp"
#include "NPY.hpp"

#include "OpticksProfile.hh"

#include "PLOG.hh"


OpticksProfile::OpticksProfile() 
   :
   m_vm0(SProc::VirtualMemoryUsageMB()),
   m_vm(NPY<float>::make(0,1,4))
{
}

void OpticksProfile::save()
{
}

void OpticksProfile::stamp(const char* tag)
{

   float vmb   = SProc::VirtualMemoryUsageMB() ; 
   float dvmb  = vmb - m_vm0 ; 
   float tsa = 0.f ; 
   float dts = 0.f ; 

   LOG(info) << "OpticksProfile::stamp " 
             << " tag " << tag
              ; 

   m_vm->add( tsa, dts, vmb, dvmb ); 
}



