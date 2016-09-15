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

::

    OpticksAnaTest --anakey tpmt --tag 10 --cat PmtInBox

**/


class OKCORE_API OpticksAna
{
    public:
       OpticksAna(Opticks* ok);
       void run();
   private:
       std::string getCommandline(const char* anakey);
       const char* getScript(const char* anakey);
       std::string getArgs(const char* anakey);
   private:
       Opticks*             m_ok ; 
       OpticksCfg<Opticks>* m_cfg ; 
       std::map<std::string, std::string> m_scripts ; 

};

#include "OKCORE_HEAD.hh"


