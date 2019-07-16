#include <sstream>

#include "SSys.hh"
#include "BResource.hh"

#include "Opticks.hh"
#include "OpticksCfg.hh"
#include "OpticksAna.hh"

#include "PLOG.hh"

const plog::Severity OpticksAna::LEVEL = debug ; 


const char* OpticksAna::DEFAULT_EXEC = "echo" ; 

OpticksAna::OpticksAna(Opticks* ok) 
   :
   m_ok(ok),
   m_cfg(ok->getCfg())
{
   // for simple filtering 
   m_scripts["tpmt"] = "tpmt.py" ;  
   m_scripts["tevt"] = "tevt.py" ;  
   m_scripts["tboolean"] = "tboolean.py" ;  
}

bool OpticksAna::isKeyEnabled(const char* anakey) const 
{
     return anakey && m_scripts.count(anakey) == 1 ;
}

const char* OpticksAna::getScript(const char* anakey) 
{
    return isKeyEnabled(anakey) ? strdup(m_scripts[anakey].c_str()) : DEFAULT_EXEC  ;  
}


std::string OpticksAna::getArgs(const char* /*anakey*/)
{
    const char* anakeyargs = m_ok->getAnaKeyArgs();
    std::stringstream ss ; 
    ss
         << "--tagoffset " << m_ok->getTagOffset() << " "
         << "--tag " << m_ok->getEventTag() << " "
         //<< "--det " << m_ok->getEventDet() << " "
         //<< "--cat " << m_ok->getInputUDet() << " "     # cat overrides not handled in python yet 
         << "--det " << m_ok->getInputUDet() << " "
         << "--pfx " << m_ok->getEventPfx() << " "
         << "--src " << m_ok->getSourceType() << " "
         << "--show " 
         << ( anakeyargs ? anakeyargs : "" )
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
   const char* anakey = m_ok->getAnaKey();
   bool enabled = isKeyEnabled(anakey) ; 
   LOG(info)
       << " anakey " << anakey  
       << " enabled " << ( enabled ? "Y" : "N" )
       ; 

   if(!enabled) return ; 

   std::string cmdline = getCommandline(anakey);

   LOG(info) << " cmdline " << cmdline ;  

   std::cout << std::endl ;  

   int rc = cmdline.empty() ? 0 : SSys::run(cmdline.c_str()) ; 

   std::cout << std::endl ;  
   
   const char* rcmsg = rc == 0 ? NULL : "OpticksAna::run non-zero RC from ana script"  ;

   int interactivity = m_ok->getInteractivityLevel() ; 

   LOG(info) 
       << " anakey " << anakey 
       << " cmdline " << cmdline
       << " interactivity " << interactivity
       << " rc " << rc
       << " rcmsg " << ( rcmsg ? rcmsg : "-" )
       ;

   if( rc != 0)  m_ok->setRC(rc, rcmsg);


   if( interactivity > 1 )
   SSys::WaitForInput("OpticksAna::run paused : hit RETURN to continue..." );    

}

