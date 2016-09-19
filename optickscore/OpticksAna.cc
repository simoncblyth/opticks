#include <sstream>

#include "SSys.hh"

#include "Opticks.hh"
#include "OpticksCfg.hh"
#include "OpticksAna.hh"

#include "PLOG.hh"

OpticksAna::OpticksAna(Opticks* ok) 
   :
   m_ok(ok),
   m_cfg(ok->getCfg())
{
   m_scripts["tpmt"] = "tpmt.py" ;  
   m_scripts["evt"] = "evt.py" ;  
}

const char* OpticksAna::getScript(const char* anakey)
{
    return m_scripts.count(anakey) == 1 ? strdup(m_scripts[anakey].c_str()) : "echo"   ;  
}

std::string OpticksAna::getArgs(const char* /*anakey*/)
{
    std::stringstream ss ; 
    ss
         << "--tag " << m_ok->getEventTag() << " "
         << "--tagoffset " << m_ok->getTagOffset() << " "
         << "--det " << m_ok->getUDet() << " "
         << "--src " << m_ok->getSourceType() << " "
         ;

    return ss.str();
}

std::string OpticksAna::getCommandline(const char* anakey)
{
    std::stringstream ss ; 
    ss
       << getScript(anakey) << " "
       << getArgs(anakey) << " "
       ;
    return ss.str();
}

void OpticksAna::run()
{
   LOG(info) << "OpticksAna::run" ; 
   std::string anakey = m_ok->getAnaKey();

   LOG(info) << "OpticksAna::run anakey " << anakey  ; 

   std::string cmdline = getCommandline(anakey.c_str());

   int rc = cmdline.empty() ? 0 : SSys::run(cmdline.c_str()) ; 

   LOG(info) << "OpticksAna::run"
             << " anakey " << anakey 
             << " cmdline " << cmdline
             << " rc " << rc
             ;

   m_ok->setRC(rc);
}

