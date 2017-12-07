#pragma once



#include <string>
#include "CFG4_API_EXPORT.hh"
#include "CFG4_HEAD.hh"

/**
CProcessManager
=================

::

   g4-;g4-cls G4ProcessManager

**/

class G4ProcessManager ;
class G4Track ;


struct CFG4_API CProcessManager
{
    static std::string Desc(G4ProcessManager* proc) ;  
    static G4ProcessManager* Current(G4Track* trk) ;  

};

#include "CFG4_TAIL.hh"
 
