#include "BFile.hh"
#include "BStr.hh"

#include "NCSG.hpp"
#include "NNodeSample.hpp"
#include "NNode.hpp"

#include "OPTICKS_LOG.hh"


void test_load_save()
{
    NCSG* csg = NCSG::Load("$TMP/tboolean-box--/1"); 
    if(!csg) return ; 
    csg->savesrc("$TMP/tboolean-box--save/1") ; 

    // savesrc after Load is an easy test to pass, as have the src buffers already from the loadsrc
}

void test_adopt_save()
{
    const char* name = "Box3" ; 
    //const char* name = "DifferenceOfSpheres" ; 

    nnode* sample = NNodeSample::Sample(name); 
    NCSG* csg = NCSG::Adopt(sample);
    assert( csg ); 

    const char* path = BStr::concat("$TMP/NCSGSaveTest/test_adopt_save/", name, NULL) ; 
    csg->savesrc(path); 


    // savesrc after Adopt is more difficult, depending on export_srcnode operation
}


const char* get_path(const char* prefix, const char* name, unsigned i )
{
    std::string path_ = BFile::FormPath(prefix, name, BStr::utoa(i) ) ; 
    const char* path = path_.c_str(); 
    return strdup(path); 
}

void test_chain()
{
    const char* name = "Box3" ; 
    nnode* sample = NNodeSample::Sample(name);
    NCSG* csg0 = NCSG::Adopt(sample);
    assert( csg0 ); 

    NCSG* csg = csg0 ; 

    const char* prefix = "$TMP/NCSGSaveTest/test_chain" ; 

    for(unsigned i=0 ; i < 5 ; i++)
    {
        const char* path = get_path( prefix, name, i ); 
        csg->savesrc(path);
        csg = NCSG::Load(path) ; 
    }
}



int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    //test_load_save(); 
    //test_adopt_save(); 
    test_chain();
 
    return 0 ; 
}


