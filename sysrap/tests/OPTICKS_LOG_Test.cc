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

#include "OPTICKS_LOG.hh"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    plog::Severity level = info ; 

    pLOG(level,4)  << " hello+4 " ; 
    pLOG(level,3)  << " hello+3 " ; 
    pLOG(level,2)  << " hello+2 " ; 
    pLOG(level,1)  << " hello+1 " ; 
    pLOG(level,0)  << " hello+0 " ; 
    pLOG(level,-1) << " hello-1 " ; 
    pLOG(level,-2) << " hello-2 " ; 
    pLOG(level,-3) << " hello-3 " ; 
    pLOG(level,-4) << " hello-4 " ; 

/*

2018-08-04 09:44:56.320 VERB  [8369891] [main@14]  hello+4 
2018-08-04 09:44:56.320 VERB  [8369891] [main@15]  hello+3 
2018-08-04 09:44:56.320 VERB  [8369891] [main@16]  hello+2 
2018-08-04 09:44:56.320 DEBUG [8369891] [main@17]  hello+1 
2018-08-04 09:44:56.320 INFO  [8369891] [main@18]  hello+0 
2018-08-04 09:44:56.320 WARN  [8369891] [main@19]  hello-1 
2018-08-04 09:44:56.320 ERROR [8369891] [main@20]  hello-2 
2018-08-04 09:44:56.320 FATAL [8369891] [main@21]  hello-3 
2018-08-04 09:44:56.320 FATAL [8369891] [main@22]  hello-4 

*/

    return 0 ; 
}

