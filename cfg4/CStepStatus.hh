#pragma once

#include <string>

#include "CFG4_PUSH.hh"
#include "G4StepStatus.hh"
#include "CFG4_POP.hh"

#include "CFG4_API_EXPORT.hh"
#include "CFG4_HEAD.hh"

struct CFG4_API CStepStatus
{
    static std::string Desc(const G4StepStatus status) ;  
};

#include "CFG4_TAIL.hh"
 


