#include <cassert>

#include "GProperty.hh"
#include "GDomain.hh"
#include "GPropertyMap.hh"

#include "PLOG.hh"
#include "GGEO_LOG.hh"



int main(int argc, char** argv)
{
    PLOG_(argc, argv);
    GGEO_LOG_ ;

    const char* matname = "FlintGlass" ;
    GPropertyMap<float>* pmap = new GPropertyMap<float>(matname);

    const char* matdir = "$TMP/opticks/GPropertyMapTest";
    pmap->save(matdir);

    GPropertyMap<float>* qmap = GPropertyMap<float>::load(matdir, matname, "material");
    assert(qmap);
    qmap->dump("qmap", 10);


    return 0 ; 
}

