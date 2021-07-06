#include <string>
#include <sstream>
#include <iostream>
#include <cassert>

#include "G4Orb.hh"
#include "G4ios.hh"

#if __clang__
     #if ((defined(G4MULTITHREADED) && !defined(G4USE_STD11)) || \
            !__has_feature(cxx_thread_local)) || !__has_feature(c_atomic)
           #define CLANG_NOSTDTLS
     #endif
#endif
 

std::string macroDesc()
{
   std::stringstream ss ; 

   #if __clang__
   ss << " __clang__ " ;  
   #else
   ss << " NOT-__clang__ " ;  
   #endif
   ss << std::endl ; 

   #if __clang__
   #if __has_feature(cxx_thread_local) 
   ss << " __clang__.cxx_thread_local " ;  
   #else
   ss << " NOT-__clang__.cxx_thread_local " ;  
   #endif
   ss << std::endl ; 

   #if __has_feature(c_atomic) 
   ss << " __clang__.c_atomic " ;  
   #else
   ss << " NOT-__clang__.c_atomic " ;  
   #endif
   ss << std::endl ; 
   #endif

   #if defined(G4MULTITHREADED) 
   ss << " G4MULTITHREADED " ;  
   #else
   ss << " NOT-G4MULTITHREADED " ;  
   #endif
   ss << std::endl ; 

   #if defined(G4USE_STD11) 
   ss << " G4USE_STD11 " ;  
   #else
   ss << " NOT-G4USE_STD11 " ;  
   #endif
   ss << std::endl ; 

   #if defined(CLANG_NOSTDTLS) 
   ss << " CLANG_NOSTDTLS " ;  
   #else
   ss << " NOT-CLANG_NOSTDTLS " ;  
   #endif
   ss << std::endl ; 

   #if defined(__INTEL_COMPILER) 
   ss << " __INTEL_COMPILER " ;  
   #else
   ss << " NOT-__INTEL_COMPILER " ;  
   #endif
   ss << std::endl ; 

   #if (defined(G4MULTITHREADED) && \
     (!defined(G4USE_STD11) || (defined(CLANG_NOSTDTLS) || defined(__INTEL_COMPILER))))
     
   ss << " G4MULTITHREADED and ( !G4USE_STD11  or CLANG_NOSTDTLS or __INTEL_COMPILER ) " ;  
   #else
   ss << " NOT [  G4MULTITHREADED and ( !G4USE_STD11  or CLANG_NOSTDTLS or __INTEL_COMPILER ) ] " ;  
   #endif
   ss << std::endl ; 

   std::string s = ss.str(); 
   return s ; 
}


int main(int , char** )
{
    double r = 100. ; 
    G4Orb orb("sphere", r );  
    assert( orb.GetRadius() == r );    
    std::cout << " orb.GetRadius() " << orb.GetRadius() << std::endl ; 

    G4cout << orb << G4endl ;   
    std::cout << macroDesc() << std::endl ; 

    return 0 ; 
}



