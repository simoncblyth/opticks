#include <cstdlib>
#include <cassert>

#include "NPY.hpp"
#include "Types.hpp"
#include "Index.hpp"

#include "PhotonsNPY.hpp"
#include "RecordsNPY.hpp"
#include "BoundariesNPY.hpp"


int main(int argc, char** argv)
{
    Types types ; 
    types.dumpFlags();

    const char* tag = "1" ;
    NPY<float>* photons = NPY<float>::load("oxcerenkov", tag,"dayabay");
    if(!photons) return 0 ;   

    NPY<short>* records = NPY<short>::load("rxcerenkov", tag,"dayabay");
    NPY<float>* domains = NPY<float>::load("domain","1","dayabay");
    NPY<int>*   idom = NPY<int>::load("idomain","1","dayabay");

    unsigned int maxrec = idom ? idom->getValue(0,0,3) : 0 ;  // TODO: enumerate the k indices 
    assert(maxrec == 10);


    const char* idpath = getenv("IDPATH");
    Index* materials = Index::load(idpath, "GMaterialIndex");

    types.setMaterialsIndex(materials);
    types.dumpMaterials();


    bool flat = false ; 
    RecordsNPY r(records, maxrec, flat);
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

