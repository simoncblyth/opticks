#pragma once
#include <cstring>

struct CustomStatus
{
   static constexpr const char* U_ = "Undefined" ; 
   static constexpr const char* F_ = "FirstPoint" ; 
   static constexpr const char* X_ = "NotCompiled" ; 
   static constexpr const char* B_ = "StandardBoundary" ; 
   static constexpr const char* N_ = "NotCustomSurfaceName" ; 
   static constexpr const char* Z_ = "NotPositiveLocalZ" ; 
   static constexpr const char* A_ = "CustomBoundaryAbsorb" ; 
   static constexpr const char* R_ = "CustomBoundaryReflect" ; 
   static constexpr const char* T_ = "CustomBoundartTransmit" ; 
   static constexpr const char* D_ = "CustomBoundaryDetect" ; 
   static constexpr const char* Y_ = "CustomARTCalc" ; 
   static const char* Name(char status); 
}; 

inline const char* CustomStatus::Name(char status)
{
   const char* s = nullptr ; 
   switch(status)
   {
       case 'U': s = U_ ; break ; 
       case 'F': s = F_ ; break ; 
       case 'X': s = X_ ; break ; 
       case 'B': s = B_ ; break ; 
       case 'N': s = N_ ; break ; 
       case 'Z': s = Z_ ; break ; 
       case 'A': s = A_ ; break ; 
       case 'R': s = R_ ; break ; 
       case 'T': s = T_ ; break ; 
       case 'D': s = D_ ; break ; 
       case 'Y': s = Y_ ; break ; 
   }
   return s ; 
}


