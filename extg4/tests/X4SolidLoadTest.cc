
#include <string>
#include "X4.hh"
#include "SSys.hh"
#include "BFile.hh"
#include "BStr.hh"
#include "NCSGList.hpp"

#include "OPTICKS_LOG.hh"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);
    
    int lvIdx = SSys::getenvint("LV",65) ;
    std::string csgpath = BFile::FormPath( X4::X4GEN_DIR, BStr::concat("x", BStr::utoa(lvIdx,3, true), NULL)) ;   

    LOG(info) << " lvIdx " << lvIdx << " csgpath " << csgpath ; 

    NCSGList* ls = NCSGList::Load(csgpath.c_str());  
    assert(ls);  
    return 0 ; 
}
