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

// TEST=BEnvTest om-t

#include <iostream>

#include "BFile.hh"
#include "BEnv.hh"
#include "BOpticksResource.hh"

#include "OPTICKS_LOG.hh"


const char* dir = "$TMP" ; 
const char* name = "G4.ini" ; 


void testCreateSave(char** envp)
{
    const char* prefix = "G4,DAE,OPTICKS,IDPATH,ENV" ; 


    BEnv* e = new BEnv(envp);

    e->dump("all");
    e->setPrefix(prefix); 
    e->dump("just prefixed");
    e->setPrefix(NULL); 
    e->dump("all again");
    e->setPrefix(prefix); 
    e->dump("just prefixed again");
    e->save(dir, name);
}


void testLoad()
{
    BEnv* f = BEnv::Load(dir, name);
    f->dump("loaded from ini");

    f->setEnvironment();
    BEnv::dumpEnvironment();
}


void testIniLoad(const char* path)
{
    //std::string fpath = BFile::FormPath(path);
    //BEnv* e = BEnv::Load(fpath.c_str());

    BEnv* e = BEnv::Load(path);

    if(!e)
    {
        LOG(error) << "MISSING " << path ;   
        return ;  
    } 

    e->dump();
    e->setEnvironment();
    BEnv::dumpEnvironment();
}



#ifdef _MSC_VER
#else
#include <unistd.h>
extern char **environ;
#endif


void testTraverse()
{
    const char* prefix = "OPTICKS_" ; 
    int i=0 ; 
    while(*(environ+i))
    {
       char* kv_ = environ[i++] ;  
       if(strncmp(kv_, prefix, strlen(prefix))==0)
       { 
           std::string kv = kv_ ; 

           size_t p = kv.find('=');  
           assert( p != std::string::npos) ; 

           std::string k = kv.substr(0,p); 
           std::string v = kv.substr(p+1);   
   
           std::cout << k << " : " << v << std::endl ;   
       }
    }      
}


void test_Create()
{
    BEnv* e = BEnv::Create("G4"); 
    assert(e); 
}
void test_getNumberOfEnvvars()
{
    BEnv* e = BEnv::Create("G4"); 
    assert(e); 
    bool require_existing_dir = true ; 
    unsigned n = e->getNumberOfEnvvars("G4", "DATA", require_existing_dir ) ; 
    LOG(info) << n ; 
}




int main(int argc, char** argv, char** /*envp*/)
{
    OPTICKS_LOG(argc, argv);

    bool testgeo(false);
    BOpticksResource rsc(testgeo) ;  // sets envvar OPTICKS_INSTALL_PREFIX internally 
    rsc.Summary();

/*
    testIniLoad("$OPTICKS_INSTALL_PREFIX/externals/config/geant4.ini") ;
    testIniLoad("$OPTICKS_INSTALL_PREFIX/opticksdata/config/opticksdata.ini") ;
    testTraverse(); 
    test_Create(); 
*/

    test_getNumberOfEnvvars(); 

    return 0 ;  
}
