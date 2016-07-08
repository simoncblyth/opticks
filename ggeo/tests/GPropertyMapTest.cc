#include <cassert>
// op --gpropertymap

#include "BOpticksResource.hh"

#include "GProperty.hh"
#include "GDomain.hh"
#include "GMaterialLib.hh"
#include "GPropertyLib.hh"
#include "GPropertyMap.hh"

#include "PLOG.hh"
#include "GGEO_LOG.hh"


int main(int argc, char** argv)
{
    PLOG_(argc, argv);
    GGEO_LOG_ ;

    BOpticksResource rsc ;  // sets internal envvar OPTICKS_INSTALL_PREFIX

    const char* path = "$OPTICKS_INSTALL_PREFIX/opticksdata/refractiveindex/tmp/glass/schott/F2.npy";
    GProperty<float>* f2 = GProperty<float>::load(path);
    if(!f2)
    {
        LOG(error) << " failed to load " << path ; 
        return 0 ;      
    } 

    assert(f2);

    f2->Summary("F2 ri", 100);

    GDomain<float>* sd = GPropertyLib::getDefaultDomain();

    const char* matname = "FlintGlass" ;

    GPropertyMap<float>* pmap = new GPropertyMap<float>(matname);

    pmap->setStandardDomain(sd);

    const char* ri = GMaterialLib::refractive_index ;

    pmap->addPropertyStandardized(ri, f2 );
   
    GProperty<float>* rip = pmap->getProperty(ri);

    rip->save("$TMP/f2.npy");

    const char* matdir = "$TMP/GPropertyMapTest";

    pmap->save(matdir);

    GPropertyMap<float>* qmap = GPropertyMap<float>::load(matdir, matname, "material");
    assert(qmap);
    qmap->dump("qmap", 10);


    return 0 ; 
}

