// name=SBacktrace_test ; gcc $name.cc -std=c++11 -lstdc++ -I.. -o /tmp/$name && /tmp/$name

#include <cassert>
#include "SBacktrace.h"

struct SBT
{
   static constexpr const char* x_summary_0 =  R"(
SBT::red
SBT::green
SBT::blue
SBT::cyan
SBT::magenta
SBT::yellow
)" ; 

   static constexpr const char* x_summary_1 =  R"(
SBT::green
SBT::blue
SBT::cyan
SBT::magenta
SBT::yellow
)" ; 

   static constexpr const char* x_summary_2 =  R"(
SBT::green
SBT::blue
SBT::cyan
SBT::yellow
)" ; 

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
   SBacktrace::DumpSummary(); 

   char* summary = SBacktrace::Summary() ; 
   bool match_0 = strstr( summary, x_summary_0 ) != nullptr ; 
   bool match_1 = strstr( summary, x_summary_1 ) != nullptr ; 
   bool match_2 = strstr( summary, x_summary_2 ) != nullptr ; 

   bool match_0s = SBacktrace::SummaryMatch(x_summary_0); 
   bool match_1s = SBacktrace::SummaryMatch(x_summary_1); 
   bool match_2s = SBacktrace::SummaryMatch(x_summary_2); 

   assert( match_0 == match_0s ); 
   assert( match_1 == match_1s ); 
   assert( match_2 == match_2s ); 

   std::cout << std::endl 
       << " match_0  " << ( match_0 ? "YES" : "NO" ) 
       << " match_1  " << ( match_1 ? "YES" : "NO" ) 
       << " match_2  " << ( match_2 ? "YES" : "NO" ) 
       <<  "[" << summary << "]" 
       << std::endl 
       ; 
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

// name=SBacktrace_test ; gcc $name.cc -std=c++11 -lstdc++ -I.. -o /tmp/$name && /tmp/$name


