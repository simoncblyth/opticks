#include "OpticksBufferControl.hh"
#include <iostream>
#include <iomanip>

#include "PLOG.hh"

typedef std::vector<const char*> VC ; 

void dump(const OpticksBufferControl& ctrl )
{
     VC tags = OpticksBufferControl::Tags();

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

     const char* ctrl_ = "OPTIX_SETSIZE,OPTIX_INPUT_OUTPUT,UPLOAD_WITH_CUDA" ;
     unsigned long long mask = OpticksBufferControl::Parse(ctrl_) ;

     std::cout << " ctrl " << ctrl_ 
               << " mask " << mask 
               << " desc " << OpticksBufferControl::Description(mask)
               << std::endl ; 


     OpticksBufferControl c0(&mask);
     dump(c0);
      

     return 0 ; 
}
