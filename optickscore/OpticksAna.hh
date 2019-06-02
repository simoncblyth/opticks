#pragma once

#include <string>
#include <map>
class Opticks ; 
template <typename T> class OpticksCfg ;

#include "OKCORE_API_EXPORT.hh"
#include "OKCORE_HEAD.hh"

/**
OpticksAna
============

Canonical m_ana OpticksAna instance is ctor resident of m_ok Opticks, 
which in turn is ctor resident of top level managers 
such as OKG4Mgr and OKMgr.

::

    OpticksAnaTest --anakey tpmt --tag 10 --cat PmtInBox

**/

class OKCORE_API OpticksAna
{
       static const char* DEFAULT_EXEC ; 
    public:
       OpticksAna(Opticks* ok);
       void run();
   private:
       std::string getCommandline(const char* anakey);
       bool isKeyEnabled(const char* anakey) const ;
       const char* getScript(const char* anakey);
       std::string getArgs(const char* anakey);
       void setEnv();
   private:
       Opticks*             m_ok ; 
       OpticksCfg<Opticks>* m_cfg ; 
       std::map<std::string, std::string> m_scripts ; 

};

#include "OKCORE_HEAD.hh"


