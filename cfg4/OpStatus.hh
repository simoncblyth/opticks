#pragma once

#include "CFG4_PUSH.hh"
#include "G4StepStatus.hh"
#include "CBoundaryProcess.hh"
#include "CStage.hh"
#include "CFG4_POP.hh"

#include "CFG4_API_EXPORT.hh"

class G4StepPoint ; 

#include <string>


class CFG4_API OpStatus 
{
    public:
        static std::string OpStepString(const G4StepStatus status);
#ifdef USE_CUSTOM_BOUNDARY
       static std::string  OpBoundaryString(const DsG4OpBoundaryProcessStatus status);
       static std::string OpBoundaryAbbrevString(const DsG4OpBoundaryProcessStatus status);
       static unsigned OpBoundaryFlag(const DsG4OpBoundaryProcessStatus status);
       static unsigned OpPointFlag(const G4StepPoint* point, const DsG4OpBoundaryProcessStatus bst, CStage::CStage_t stage);
#else
       static std::string  OpBoundaryString(const G4OpBoundaryProcessStatus status);
       static std::string OpBoundaryAbbrevString(const G4OpBoundaryProcessStatus status);
       static unsigned OpBoundaryFlag(const G4OpBoundaryProcessStatus status);
       static unsigned OpPointFlag(const G4StepPoint* point, const G4OpBoundaryProcessStatus bst, CStage::CStage_t stage);
#endif

       static  bool IsTerminalFlag(unsigned flag); 
};



