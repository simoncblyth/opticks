#include <cstdlib>
#include <cassert>

#include "NPY.hpp"
#include "Types.hpp"
#include "Index.hpp"

#include "PhotonsNPY.hpp"
#include "RecordsNPY.hpp"
#include "BoundariesNPY.hpp"

#include "OPTICKS_LOG.hh"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    Types types ; 
    types.dumpFlags();

    const char* pfx = "tboolean-proxy-11" ;   
    const char* typ = "torch" ;
    const char* tag = "1" ;
    const char* det = pfx ;

    NPY<float>* photons = NPY<float>::load(pfx, "ox%s", typ, tag, det);
    if(!photons) return 0 ;   

    NPY<short>* records = NPY<short>::load(pfx, "rx%s",   typ, tag, det);
    NPY<float>* domains = NPY<float>::load(pfx, "fdom%s", typ, tag, det);
    NPY<int>*   idom    =   NPY<int>::load(pfx, "idom%s", typ, tag, det);

    if(idom == NULL)
    {  
       LOG(warning) << "FAILED TO LOAD idom " ;
    }

    unsigned int maxrec = idom ? idom->getValue(0,0,3) : 0 ;  // TODO: enumerate the k indices 
    
    if(maxrec != 10)
    {
       LOG(fatal) << "UNEXPECTED maxrec " << maxrec ;   
    }
    assert(maxrec == 10);


    const char* idpath = getenv("IDPATH");
    const char* reldir = NULL ; 
    Index* materials = Index::load(idpath, "GMaterialIndex", reldir);

    types.setMaterialsIndex(materials);
    types.dumpMaterials();


    RecordsNPY r(records, maxrec);
    r.setTypes(&types);
    r.setDomains(domains);

    PhotonsNPY p(photons);
    p.setTypes(&types);
    p.setRecs(&r);


    if(argc > 1)
    {
       for(int i=0 ; i < argc-1 ; i++) p.dump(atoi(argv[i+1]));
    }
    else
    {
       p.dumpPhotons("photons", 30);
    
       NPY<float>* pathinfo = p.make_pathinfo();
       pathinfo->save("$TMP/pathinfo.npy");
    }



    return 0 ;
}

