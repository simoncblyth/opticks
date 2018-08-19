
#include "SBacktrace.hh"
#include "OPTICKS_LOG.hh"


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
   LOG(info) << "." ; 
   SBacktrace::Dump(); 
}
void SBT::green(){  
   LOG(info) << "." ; 
   red(); 
}
void SBT::blue(){  
   LOG(info) << "." ; 
   green(); 
}
void SBT::cyan(){  
   LOG(info) << "." ; 
   blue(); 
}
void SBT::magenta(){  
   LOG(info) << "." ; 
   cyan(); 
}
void SBT::yellow(){  
   LOG(info) << "." ; 
   magenta(); 
}

int main(int argc, char** argv)
{  
    OPTICKS_LOG(argc, argv); 
    SBT::yellow();      
    return 0 ; 
}
