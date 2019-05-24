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



int main(int argc, char** argv, char** /*envp*/)
{
    OPTICKS_LOG(argc, argv);

    BOpticksResource rsc ;  // sets envvar OPTICKS_INSTALL_PREFIX internally 
    rsc.Summary();

/*
    testIniLoad("$OPTICKS_INSTALL_PREFIX/externals/config/geant4.ini") ;
    testIniLoad("$OPTICKS_INSTALL_PREFIX/opticksdata/config/opticksdata.ini") ;
*/
    testTraverse(); 


    return 0 ;  
}
