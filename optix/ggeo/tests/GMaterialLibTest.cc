// ggv --mat

#include "GCache.hh"
#include "GMaterialLib.hh"
#include "GAttrSeq.hh"

#include <string>
#include <iostream>
#include <ostream>   
#include <algorithm>
#include <iterator>
#include <iomanip>



void colorcodes(GMaterialLib* mlib)
{
    std::vector<unsigned int> cc = mlib->getAttrNames()->getColorCodes();

    //std::copy( cc.begin(), cc.end(), std::ostream_iterator<unsigned int>(std::cout, "\n"));

    for(unsigned int i=0 ; i < cc.size() ; i++)
    {
       std::cout << std::setw(5) << i 
                 << std::setw(10) << std::hex << cc[i] << std::dec
                 << std::endl ; 
    }
}


void attrdump( GMaterialLib* mlib)
{
    const char* mats = "Acrylic,GdDopedLS,LiquidScintillator,ESR,MineralOil" ;

    GAttrSeq* amat = mlib->getAttrNames();

    amat->dump(mats);
}


void test_getMaterial(GMaterialLib* mlib)
{
    const char* name = "Water" ;
    GMaterial* mat = mlib->getMaterial(name);
    mlib->dump(mat, name);
}



int main(int argc, char** argv)
{
    GCache gc("GGEOVIEW_","mats.log", "info");
    gc.configure(argc, argv);

    GMaterialLib* mlib = GMaterialLib::load(&gc);
    mlib->dump();

    return 0 ;
}

