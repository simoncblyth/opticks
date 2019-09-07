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


// optickscore/tests/OpticksResourceTest.cc
// op --resource
// op --j1707 --resource

#include "Opticks.hh"
#include "OpticksResource.hh"


#include "OPTICKS_LOG.hh"

void dumpenv_0(char** envp)
{
   // http://stackoverflow.com/questions/2085302/printing-all-environment-variables-in-c-c
    char** env;
    for (env = envp; *env != 0; env++)
    {
       char* thisEnv = *env;
       printf("%s\n", thisEnv);    
    }
}


void dumpenv_1(char** envp)
{
    while(*envp) printf("%s\n",*envp++);
}



int main(int argc, char** argv, char** /*envp*/)
{
    OPTICKS_LOG(argc, argv);

    Opticks ok(argc, argv) ;
    ok.configure();       // <-- resource setup now done in here 

    LOG(info) << " ok.getExampleMaterialNames() " << ok.getExampleMaterialNames() ;

/*
   resource setup is now 
    OpticksResource res(&ok) ;  // TODO: remove duplication of envprefix beween both these
    res.Dump();
*/

    OpticksResource* res = ok.getResource(); 
    res->Dump(); 

    return 0 ; 
}
