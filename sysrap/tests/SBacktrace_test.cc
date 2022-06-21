// name=SBacktrace_test ; gcc $name.cc -std=c++11 -lstdc++ -I.. -o /tmp/$name && /tmp/$name

#include "SBacktrace.h"

struct SBT
{
   static void red(); 
   static void green(); 
   static void blue(); 
   static void cyan(); 
   static void magenta(); 
   static void yellow(); 

};

void SBT::red(){  
   std::cerr << "red" << std::endl ; 
   SBacktrace::Dump(); 
}
void SBT::green(){  
   std::cerr << "green" << std::endl  ; 
   red(); 
}
void SBT::blue(){  
   std::cerr << "blue" << std::endl  ; 
   green(); 
}
void SBT::cyan(){  
   std::cerr << "cyan" << std::endl ; 
   blue(); 
}
void SBT::magenta(){  
   std::cerr << "magenta" << std::endl  ; 
   cyan(); 
}
void SBT::yellow(){  
   std::cerr << "yellow" << std::endl ; 
   magenta(); 
}

int main(int argc, char** argv)
{  
    SBT::yellow();      
    return 0 ; 
}
