#include "BFile.hh"
#include "BRegex.hh"

int main()
{
    std::string path = BFile::FormPath("~", "env/optickscore", "OpticksPhoton.h" );
    std::cout << "extract enum pairs from file " << path << std::endl ;

    BRegex::upairs_t upairs ; 
    BRegex::enum_regexsearch( upairs, path.c_str() );
    BRegex::udump(upairs);

    return 0 ; 
}
