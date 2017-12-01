
#include "PLOG.hh"

#define G4DAE_EXTRAS_NO_VALUE
#define G4DAE_EXTRAS_WITH_ONE 1
#define G4DAE_EXTRAS_WITH_ZERO 0

int main(int argc, char** argv)
{
   PLOG_(argc, argv);

#ifdef G4DAE_EXTRAS_NO_VALUE
   LOG(info) << "G4DAE_EXTRAS_NO_VALUE" ; 
#else
   LOG(info) << "not G4DAE_EXTRAS_NO_VALUE" ; 
#endif   

#ifdef G4DAE_EXTRAS_WITH_ONE
   LOG(info) << "G4DAE_EXTRAS_WITH_ONE" ; 
#else
   LOG(info) << "not G4DAE_EXTRAS_WITH_ONE" ; 
#endif   


#ifdef G4DAE_EXTRAS_WITH_ZERO
   LOG(info) << "G4DAE_EXTRAS_WITH_ZERO" ; 
#else
   LOG(info) << "not G4DAE_EXTRAS_WITH_ZERO" ; 
#endif   


   return 0 ; 
}


/*

simon:sysrap blyth$ uname -a
Darwin simon.phys.ntu.edu.tw 13.3.0 Darwin Kernel Version 13.3.0: Tue Jun  3 21:27:35 PDT 2014; root:xnu-2422.110.17~1/RELEASE_X86_64 x86_64
simon:sysrap blyth$ hash_define_without_value 
2017-12-01 11:11:17.313 INFO  [777934] [main@13] G4DAE_EXTRAS_NO_VALUE
2017-12-01 11:11:17.313 INFO  [777934] [main@19] G4DAE_EXTRAS_WITH_ONE
2017-12-01 11:11:17.313 INFO  [777934] [main@26] G4DAE_EXTRAS_WITH_ZERO
simon:sysrap blyth$ 


*/



