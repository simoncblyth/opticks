#include "OpticksActionControl.hh"
#include <iostream>
#include <iomanip>

#include "PLOG.hh"

typedef std::vector<const char*> VC ; 

void dump(const OpticksActionControl& ctrl, const char* msg="")
{
     LOG(info) << msg ; 

     VC tags = OpticksActionControl::Tags();

     for(VC::const_iterator it=tags.begin() ; it != tags.end() ; it++)
     {
         const char* tag = *it ; 
         bool set = ctrl.isSet(tag) ;

         LOG(info) << std::setw(20) << tag 
                   << " " << ( set ? "Y" : "N" )
                   ;
     } 
}


int main(int argc, char** argv)
{
     PLOG_(argc, argv);

     const char* ctrl_ = "GS_LOADED,GS_FABRICATED,GS_TRANSLATED" ;
     unsigned long long mask = OpticksActionControl::Parse(ctrl_) ;

     std::cout << " ctrl " << ctrl_ 
               << " mask " << mask 
               << " desc " << OpticksActionControl::Description(mask)
               << std::endl ; 


     OpticksActionControl c0(&mask);
     dump(c0);

     const char* label = "GS_TORCH" ; 

     c0.add(label);
     assert(c0.isSet(label));

     dump(c0, "after addition");
      
 

     return 0 ; 
}
