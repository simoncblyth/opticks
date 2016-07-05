// ggv --mat

#include <string>
#include <iostream>
#include <ostream>   
#include <algorithm>
#include <iterator>
#include <iomanip>


#include "SSys.hh"

#include "Opticks.hh"
#include "OpticksAttrSeq.hh"

#include "GProperty.hh"
#include "GPropertyMap.hh"
#include "GMaterial.hh"
#include "GMaterialLib.hh"

#include "PLOG.hh"
#include "GGEO_LOG.hh"



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

    OpticksAttrSeq* amat = mlib->getAttrNames();

    amat->dump(mats);
}


void test_getMaterial(GMaterialLib* mlib)
{
    const char* name = "Water" ;
    GMaterial* mat = mlib->getMaterial(name);
    mlib->dump(mat, name);
}


void test_addTestMaterial(GMaterialLib* mlib)
{
    // see GGeo::addTestMaterials
    //
    // Moved to opticksdata for portability  .,$s,LOCAL_BASE/env/physics,OPTICKS_INSTALL_PREFIX/opticksdata,g 

    GProperty<float>* f2 = GProperty<float>::load("$OPTICKS_INSTALL_PREFIX/opticksdata/refractiveindex/tmp/glass/schott/F2.npy");
    if(f2)
    {
        f2->Summary("F2 ri", 100);
        GMaterial* raw = new GMaterial("GlassSchottF2", mlib->getNumMaterials() );
        raw->addPropertyStandardized( GMaterialLib::refractive_index_local, f2 ); 

        mlib->setClosed(false);  // OK for testing only 
        mlib->add(raw);
    }
}



int main(int argc, char** argv)
{
    PLOG_(argc, argv);
    GGEO_LOG_ ;

    SSys::setenvvar("", "IDPATH", "/tmp", true );

    Opticks ok(argc, argv);

    LOG(info) << " ok " ; 

    GMaterialLib* mlib = GMaterialLib::load(&ok);

    LOG(info) << " after load " ; 
    test_addTestMaterial(mlib);

    mlib->dump();
    LOG(info) << " after dump " ; 

    return 0 ;
}

