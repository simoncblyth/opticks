#include "regexsearch.hh"
#include <iomanip>


int main(int argc, char** argv)
{
    const char* default_ptn = "<[^>]*>";
    const char* ptn =  argc > 1 ? argv[1] : default_ptn ;  

    std::cout << "search cin for text matching regex " << ptn << std::endl ;

    boost::regex e(ptn);

    pairs_t pairs ; 
    regexsearch( pairs, std::cin , e );
    dump(pairs, "pairs plucked using regexp");


    return 0 ; 
}
