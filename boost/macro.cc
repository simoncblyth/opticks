


#include <boost/predef.h>

#include <iostream>

int main()
{

#if defined BOOST_OS_MACOS
   std::cout << "BOOST_OS_MACOS" << std::endl ; 
#elif defined BOOST_OS_WINDOWS
   std::cout << "BOOST_OS_WINDOWS" << std::endl ; 
#elif defined BOOST_OS_LINUX
   std::cout << "BOOST_OS_LINUX" << std::endl ; 
#endif


#if defined _MSC_VER
   std::cout << "MSC_VER" << std::endl ; 
#elif defined(__MINGW32__)
   std::cout << "MIGW32" << std::endl ; 
#elif defined(__APPLE__)
   std::cout << "APPLE" << std::endl ; 
#endif


   return 0 ;
}

