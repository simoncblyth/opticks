/*
 * Copyright (c) 2019 Opticks Team. All Rights Reserved.
 *
 * This file is part of Opticks
 * (see https://bitbucket.org/simoncblyth/opticks).
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); 
 * you may not use this file except in compliance with the License.  
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software 
 * distributed under the License is distributed on an "AS IS" BASIS, 
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
 * See the License for the specific language governing permissions and 
 * limitations under the License.
 */



#include <cassert>
#include "PLOG.hh"

#define G4DAE_EXTRAS_NO_VALUE
#define G4DAE_EXTRAS_WITH_ONE 1
#define G4DAE_EXTRAS_WITH_ZERO 0

int main(int argc, char** argv)
{
   PLOG_(argc, argv);

   int count(0); 

#ifdef G4DAE_EXTRAS_NO_VALUE
   LOG(info) << "G4DAE_EXTRAS_NO_VALUE" ; 
   count++ ; 
#else
   LOG(info) << "not G4DAE_EXTRAS_NO_VALUE" ; 
#endif   

#ifdef G4DAE_EXTRAS_WITH_ONE
   LOG(info) << "G4DAE_EXTRAS_WITH_ONE" ; 
   count++ ; 
#else
   LOG(info) << "not G4DAE_EXTRAS_WITH_ONE" ; 
#endif   


#ifdef G4DAE_EXTRAS_WITH_ZERO
   LOG(info) << "G4DAE_EXTRAS_WITH_ZERO" ; 
   count++ ; 
#else
   LOG(info) << "not G4DAE_EXTRAS_WITH_ZERO" ; 
#endif   


   LOG(info) << " count : " << count ; 
   assert(count == 3 );

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



