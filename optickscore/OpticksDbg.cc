#include <sstream>
#include <algorithm>

#include "BStr.hh"

#include "Opticks.hh"
#include "OpticksCfg.hh"
#include "OpticksDbg.hh"

#include "PLOG.hh"

OpticksDbg::OpticksDbg(Opticks* ok) 
   :
   m_ok(ok),
   m_cfg(NULL)
{
}

void OpticksDbg::postconfigure()
{
   LOG(trace) << "setting up"  ; 
   m_cfg = m_ok->getCfg();

   const std::string& dindex = m_cfg->getDbgIndex() ;
   if(dindex.empty())
   {
       LOG(trace) << "empty" ;
   } 
   else
   { 
       LOG(trace) << " dindex " << dindex ;  
       BStr::isplit(m_debug_photon, dindex.c_str(), ',');
   }

   LOG(info) << "OpticksDbg::postconfigure" << description() ; 
}


bool OpticksDbg::isDbgPhoton(int photon_id)
{
    return std::find(m_debug_photon.begin(), m_debug_photon.end(), photon_id ) != m_debug_photon.end() ; 
}


std::string OpticksDbg::description()
{
    std::stringstream ss ; 
    ss << " OpticksDbg debug_photon "
       << " size: " << m_debug_photon.size()
       << " elem: (" << BStr::ijoin(m_debug_photon, ',') << ")" 
       ;
    return ss.str(); 
}


const std::vector<int>&  OpticksDbg::getDbgIndex()
{
   return m_debug_photon ;
}


