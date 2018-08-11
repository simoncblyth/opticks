
#include <cassert>
#include "CFG4_BODY.hh"
#include "G4String.hh"

#include "OPTICKS_LOG.hh"


void test_remove()
{
    G4String name = "/some/path/to/a/file.txt" ; 

    size_t sLast = name.last('/');


    G4String SensitiveDetectorName ;
    G4String thePathName ; 
    G4String fullPathName  ;

    if(sLast==std::string::npos)
    { // detector name only
        SensitiveDetectorName = name;
        thePathName = "/";
    }
    else
    { // name conatin the directory path
        SensitiveDetectorName = name;
        SensitiveDetectorName.remove(0,sLast+1);
        thePathName = name;
        thePathName.remove(sLast+1,name.length()-sLast);
        if(thePathName(0)!='/') thePathName.prepend("/");
    }
    fullPathName = thePathName + SensitiveDetectorName;
     
    LOG(info) 
       << std::endl 
       << " name " << name << std::endl 
       << " SensitiveDetectorName " << SensitiveDetectorName << std::endl
       << " thePathName " << thePathName << std::endl
       << " fullPathName " << fullPathName << std::endl
       ;

}





int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    test_remove();  


    return 0 ;
}

