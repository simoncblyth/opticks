
#include "OPTICKS_LOG.hh"
#include "ssys.h"
#include "spath.h"
#include "scuda.h"
#include "NP.hh"
#include "NPFold.h"

#include "QScint.hh"

struct QScintTest
{
    const QScint* scint ;
    const NP*     src ;
    const NP*     dst ;

    QScintTest( const QScint* scint );
    void save(const char* base);
};


inline QScintTest::QScintTest( const QScint* scint_ )
    :
    scint(scint_),
    src(scint->src),
    dst(scint->lookup())
{
}

inline void QScintTest::save(const char* base)
{
    NPFold* fold = new NPFold ;
    fold->add("src", src);
    fold->add("dst", dst);
    fold->save(base);
}



int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    const char* path = spath::Resolve("$CFBaseFromGEOM/CSGFoundry/SSim/stree/standard/icdf.npy");
    NP* icdf = NP::LoadIfExists(path);

    LOG(info)
        << " path " << path
        << " icdf " << ( icdf ? icdf->sstr() : "-" )
        ;

    if(icdf == nullptr) return 0 ;

    unsigned hd_factor = 0u ;
    QScint sc(icdf, hd_factor);     // uploads reemission texture

    //sc.check();
    QScintTest t(&sc);
    t.save("$FOLD");


    return 0 ;
}

