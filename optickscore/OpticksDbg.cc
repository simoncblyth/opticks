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
   const std::string& oindex = m_cfg->getOtherIndex() ;

   if(dindex.empty())
   {
       LOG(trace) << "dindex empty" ;
   } 
   else
   { 
       LOG(trace) << " dindex " << dindex ;  
       BStr::isplit(m_debug_photon, dindex.c_str(), ',');
   }


   if(oindex.empty())
   {
       LOG(trace) << "oindex empty" ;
   } 
   else
   { 
       LOG(trace) << " oindex " << oindex ;  
       BStr::isplit(m_other_photon, oindex.c_str(), ',');
   }


   LOG(debug) << "OpticksDbg::postconfigure" << description() ; 
}



bool OpticksDbg::isDbgPhoton(int record_id)
{
    return std::find(m_debug_photon.begin(), m_debug_photon.end(), record_id ) != m_debug_photon.end() ; 
}
bool OpticksDbg::isOtherPhoton(int record_id)
{
    return std::find(m_other_photon.begin(), m_other_photon.end(), record_id ) != m_other_photon.end() ; 
}




std::string OpticksDbg::description()
{
    std::stringstream ss ; 
    ss << " OpticksDbg "
       << " debug_photon "
       << " size: " << m_debug_photon.size()
       << " elem: (" << BStr::ijoin(m_debug_photon, ',') << ")" 
       << " other_photon "
       << " size: " << m_other_photon.size()
       << " elem: (" << BStr::ijoin(m_other_photon, ',') << ")" 
       ;
    return ss.str(); 
}


const std::vector<int>&  OpticksDbg::getDbgIndex()
{
   return m_debug_photon ;
}
const std::vector<int>&  OpticksDbg::getOtherIndex()
{
   return m_other_photon ;
}

