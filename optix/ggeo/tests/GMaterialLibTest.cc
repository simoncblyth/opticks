#include "GCache.hh"
#include "GMaterialLib.hh"

#include <string>
#include <iostream>
#include <ostream>   
#include <algorithm>
#include <iterator>
#include <iomanip>


int main()
{
    GCache gc("GGEOVIEW_");

    GMaterialLib* lib = GMaterialLib::load(&gc);

    const char* mats = "Acrylic,GdDopedLS,LiquidScintillator,ESR,MineralOil" ;

    lib->dumpItems(mats);

    std::vector<unsigned int> cc = lib->getColorCodes();

    //std::copy( cc.begin(), cc.end(), std::ostream_iterator<unsigned int>(std::cout, "\n"));

    for(unsigned int i=0 ; i < cc.size() ; i++)
    {

       std::cout << std::setw(5) << i 
                 << std::setw(10) << std::hex << cc[i] << std::dec
                 << std::endl ; 
    }



    return 0 ;
}

