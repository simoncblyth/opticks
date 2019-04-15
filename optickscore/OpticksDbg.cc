#include <cassert>
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
   m_cfg(NULL),
   m_mask_buffer(NULL)
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
unsigned OpticksDbg::getNumMaskPhoton() const 
{
    return m_mask.size() ; 
}
unsigned OpticksDbg::getNumX4PolySkip() const 
{
    return m_x4polyskip.size() ; 
}
unsigned OpticksDbg::getNumCSGSkipLV() const 
{
    return m_csgskiplv.size() ; 
}





NPY<unsigned>* OpticksDbg::getMaskBuffer() const
{
    return m_mask_buffer ; 
}

unsigned OpticksDbg::getMaskIndex(unsigned idx) const 
{
    assert( idx < m_mask.size() );
    return m_mask[idx] ; 
}


const std::vector<unsigned>&  OpticksDbg::getMask()
{
   return m_mask ;
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

    LOG(verbose) << "loaded " << vec.size() << " from " << path ; 
    assert( vec.size() == u->getShape(0) ); 
}



void OpticksDbg::postconfigure()
{
   LOG(verbose) << "setting up"  ; 
   m_cfg = m_ok->getCfg();

   const std::string& dindex = m_cfg->getDbgIndex() ;
   const std::string& oindex = m_cfg->getOtherIndex() ;

   const std::string& mask = m_cfg->getMask() ;
   const std::string& x4polyskip = m_cfg->getX4PolySkip() ;
   const std::string& csgskiplv = m_cfg->getCSGSkipLV() ;

   postconfigure( dindex, m_debug_photon );
   postconfigure( oindex, m_other_photon );
   postconfigure( mask, m_mask );
   postconfigure( x4polyskip, m_x4polyskip );
   postconfigure( csgskiplv, m_csgskiplv );

   if(m_mask.size() > 0)
   {
       m_mask_buffer = NPY<unsigned>::make_from_vec(m_mask); 
   } 

   LOG(debug) << "OpticksDbg::postconfigure" << description() ; 
}




void OpticksDbg::postconfigure(const std::string& spec, std::vector<unsigned>& ls)
{
   if(spec.empty())
   {
       LOG(verbose) << "spec empty" ;
   } 
   else if(BFile::LooksLikePath(spec.c_str()))
   { 
       loadNPY1(ls, spec.c_str() );
   }
   else
   { 
       LOG(verbose) << " spec " << spec ;  
       BStr::usplit(ls, spec.c_str(), ',');
   }
}





bool OpticksDbg::isDbgPhoton(unsigned record_id) const 
{
    return std::find(m_debug_photon.begin(), m_debug_photon.end(), record_id ) != m_debug_photon.end() ; 
}
bool OpticksDbg::isOtherPhoton(unsigned record_id) const 
{
    return std::find(m_other_photon.begin(), m_other_photon.end(), record_id ) != m_other_photon.end() ; 
}
bool OpticksDbg::isMaskPhoton(unsigned record_id) const
{
    return std::find(m_mask.begin(), m_mask.end(), record_id ) != m_mask.end() ; 
}
bool OpticksDbg::isX4PolySkip(unsigned lvIdx) const 
{
    return std::find(m_x4polyskip.begin(), m_x4polyskip.end(), lvIdx ) != m_x4polyskip.end() ; 
}
bool OpticksDbg::isCSGSkipLV(unsigned lvIdx) const 
{
    return std::find(m_csgskiplv.begin(), m_csgskiplv.end(), lvIdx ) != m_csgskiplv.end() ; 
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


