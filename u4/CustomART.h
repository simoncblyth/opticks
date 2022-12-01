#pragma once
#include <cstring>

struct CustomART_
{
   static constexpr const char* ACTIVE = "ARTD" ; 
   static constexpr const char* U_ = "Undefined" ; 
   static constexpr const char* N_ = "NameOfOpticalSurfaceUnmatched" ; 
   static constexpr const char* Z_ = "ZLocalDoesNotTrigger" ; 
   static constexpr const char* A_ = "Absorb" ; 
   static constexpr const char* R_ = "Reflect" ; 
   static constexpr const char* T_ = "Transmit" ; 
   static constexpr const char* D_ = "Detect" ; 

   static bool IsActive(char status) ; 
   static const char* Desc(char status); 
}; 

inline bool CustomART_::IsActive(char status)
{
    return strchr(ACTIVE, status) != nullptr ; 
} 

inline const char* CustomART_::Desc(char status)
{
   const char* s = nullptr ; 
   switch(status)
   {
       case 'U': s = U_ ; break ; 
       case 'N': s = N_ ; break ; 
       case 'Z': s = Z_ ; break ; 
       case 'A': s = A_ ; break ; 
       case 'R': s = R_ ; break ; 
       case 'T': s = T_ ; break ; 
       case 'D': s = D_ ; break ; 
   }
   return s ; 
}

