#include "GProperty.hh"
#include "GDomain.hh"
#include "GMaterialLib.hh"
#include "GPropertyLib.hh"
#include "GPropertyMap.hh"

int main(int argc, char** argv)
{
    GProperty<float>* f2 = GProperty<float>::load("$LOCAL_BASE/env/physics/refractiveindex/tmp/glass/schott/F2.npy");

    f2->Summary("F2 ri", 100);

    GDomain<float>* sd = GPropertyLib::getDefaultDomain();

    GPropertyMap<float>* pmap = new GPropertyMap<float>("FlintGlass");

    pmap->setStandardDomain(sd);

    const char* ri = GMaterialLib::refractive_index ;

    pmap->addPropertyStandardized(ri, f2 );
   
    GProperty<float>* rip = pmap->getProperty(ri);

    rip->save("/tmp/f2.npy");



    return 0 ; 
}

