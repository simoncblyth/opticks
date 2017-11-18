#include <sstream>
#include <algorithm>

#include "BFile.hh"
#include "BStr.hh"
#include "NPY.hpp"

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


unsigned OpticksDbg::getNumDbgPhoton() const 
{
    return m_debug_photon.size() ; 
}

unsigned OpticksDbg::getNumOtherPhoton() const 
{
    return m_other_photon.size() ; 
}


void OpticksDbg::loadNPY1(std::vector<unsigned>& vec, const char* path )
{
    NPY<unsigned>* u = NPY<unsigned>::load(path) ;
    if(!u) 
    {
       LOG(warning) << " failed to load " << path ; 
       return ; 
    } 
    assert( u->hasShape(-1) );
    //u->dump();

    u->copyTo(vec);

    LOG(trace) << "loaded " << vec.size() << " from " << path ; 
    assert( vec.size() == u->getShape(0) ); 
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
   else if(BFile::LooksLikePath(dindex.c_str()))
   { 
       LOG(trace) << "dindex LooksLikePath "  << dindex ;
       loadNPY1(m_debug_photon, dindex.c_str() );
   }
   else
   { 
       LOG(trace) << " dindex " << dindex ;  
       BStr::usplit(m_debug_photon, dindex.c_str(), ',');
   }


   if(oindex.empty())
   {
       LOG(trace) << "oindex empty" ;
   } 
   else if(BFile::LooksLikePath(oindex.c_str()))
   { 
       loadNPY1(m_other_photon, oindex.c_str() );
   }
   else
   { 
       LOG(trace) << " oindex " << oindex ;  
       BStr::usplit(m_other_photon, oindex.c_str(), ',');
   }


   LOG(debug) << "OpticksDbg::postconfigure" << description() ; 
}



bool OpticksDbg::isDbgPhoton(unsigned record_id)
{
    return std::find(m_debug_photon.begin(), m_debug_photon.end(), record_id ) != m_debug_photon.end() ; 
}
bool OpticksDbg::isOtherPhoton(unsigned record_id)
{
    return std::find(m_other_photon.begin(), m_other_photon.end(), record_id ) != m_other_photon.end() ; 
}




std::string OpticksDbg::description()
{
    std::stringstream ss ; 
    ss << " OpticksDbg "
       << " debug_photon "
       << " size: " << m_debug_photon.size()
       << " elem: (" << BStr::ujoin(m_debug_photon, ',') << ")" 
       << " other_photon "
       << " size: " << m_other_photon.size()
       << " elem: (" << BStr::ujoin(m_other_photon, ',') << ")" 
       ;
    return ss.str(); 
}


const std::vector<unsigned>&  OpticksDbg::getDbgIndex()
{
   return m_debug_photon ;
}
const std::vector<unsigned>&  OpticksDbg::getOtherIndex()
{
   return m_other_photon ;
}

