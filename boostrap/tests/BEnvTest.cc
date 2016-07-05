#include <iostream>

#include "BFile.hh"
#include "BEnv.hh"
#include "BOpticksResource.hh"

#include "PLOG.hh"
#include "SYSRAP_LOG.hh"
#include "BRAP_LOG.hh"


const char* dir = "/tmp" ; 
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
    BEnv* f = BEnv::load(dir, name);
    f->dump("loaded from ini");

    f->setEnvironment();
    BEnv::dumpEnvironment();
}


void testIniLoad(const char* path)
{
    //std::string fpath = BFile::FormPath(path);
    //BEnv* e = BEnv::load(fpath.c_str());

    BEnv* e = BEnv::load(path);


    if(!e)
    {
        LOG(error) << "MISSING " << path ;   
        return ;  
    } 

    e->dump();
    e->setEnvironment();
    BEnv::dumpEnvironment();
}


int main(int argc, char** argv, char** /*envp*/)
{
    PLOG_(argc, argv);

    BRAP_LOG__ ; 
    SYSRAP_LOG__ ; 

    BOpticksResource rsc ;  // sets envvar OPTICKS_INSTALL_PREFIX internally 
    rsc.Summary();

    testIniLoad("$OPTICKS_INSTALL_PREFIX/externals/config/geant4.ini") ;
    testIniLoad("$OPTICKS_INSTALL_PREFIX/opticksdata/config/opticksdata.ini") ;

    return 0 ;  
}
