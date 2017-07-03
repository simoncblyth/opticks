#include <cstdlib>
#include <cassert>

#include "NPY.hpp"
#include "Types.hpp"
#include "Index.hpp"

#include "PhotonsNPY.hpp"
#include "RecordsNPY.hpp"
#include "BoundariesNPY.hpp"

#include "PLOG.hh"

int main(int argc, char** argv)
{
    PLOG_(argc, argv);

    Types types ; 
    types.dumpFlags();

    const char* typ = "cerenkov" ;
    const char* tag = "1" ;
    const char* det = "dayabay" ;

    NPY<float>* photons = NPY<float>::load("ox%s", typ, tag, det);
    if(!photons) return 0 ;   

    NPY<short>* records = NPY<short>::load("rx%s",   typ, tag, det);
    NPY<float>* domains = NPY<float>::load("fdom%s", typ, tag, det);
    NPY<int>*   idom    =   NPY<int>::load("idom%s", typ, tag, det);

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

