#include <iostream>

#include "BFile.hh"
#include "NEnv.hpp"

#include "PLOG.hh"
#include "NPY_LOG.hh"



const char* dir = "/tmp" ; 
const char* name = "G4.ini" ; 



void testCreateSave(char** envp)
{
    const char* prefix = "G4,DAE,OPTICKS,IDPATH,ENV" ; 


    NEnv* e = new NEnv(envp);

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
    NEnv* f = NEnv::load(dir, name);
    f->dump("loaded from ini");

    f->setEnvironment();
    NEnv::dumpEnvironment();
}



int main(int argc, char** argv, char** /*envp*/)
{
    PLOG_(argc, argv);

    NPY_LOG__ ; 

    const char* path =  "$LOCAL_BASE/opticks/externals/config/geant4.ini" ;
    NEnv* e = NEnv::load(path);


    e->dump();
    e->setEnvironment();
    NEnv::dumpEnvironment();


    return 0 ;  
}
