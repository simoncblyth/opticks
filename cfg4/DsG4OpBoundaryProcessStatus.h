#pragma once

namespace Ds {

enum DsG4OpBoundaryProcessStatus 
{  
     Undefined,
     FresnelRefraction, 
     FresnelReflection,
     TotalInternalReflection,
     LambertianReflection, 
     LobeReflection,
     SpikeReflection, 
     BackScattering,
     Absorption, 
     Detection, 
     NotAtBoundary,
     SameMaterial, 
     StepTooSmall, 
     NoRINDEX 
};


}

