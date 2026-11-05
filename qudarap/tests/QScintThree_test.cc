/**
QScintThree_test.sh
====================

~/o/qudarap/tests/QScintThree_test.sh

**/

#include "spath.h"
#include "ssys.h"
#include "U4ScintThree.h"
#include "U4ScintThreeExtra.h"
#include "QScintThree.h"

struct QScintThree_test
{
    bool               hd ;
    size_t             num_wlsamp ;
    const QScintThree* qs ;
    const NP*          wl ;

    QScintThree_test(const QScintThree* qs);
    void save(const char* base) const ;
};

inline QScintThree_test::QScintThree_test(const QScintThree* qs_)
    :
    hd(ssys::getenvbool("Q4ScintThree__SD") == false),
    num_wlsamp(ssys::getenvsizespec("Q4ScintThree__num_wlsamp", "M100")),
    qs(qs_),
    wl(qs ? qs->wavelength(2, num_wlsamp, hd) : nullptr)  // 2 emitting species: PPO, bisMSB (LAB does not emit)
{
}

inline void QScintThree_test::save(const char* base) const
{
    NPFold* f = new NPFold ;
    f->set_meta<int>("HD", int(hd));
    f->add("wl", wl);
    f->save(base);
}


/**
QScintThree_test.cc:main
--------------------------

1. load material properties from Opticks geometry
2. invoke U4ScintThree::Create which does standard Geant4 wavelength generation and prepares (icdf_three/const_three) - save to $FOLD/U4ScintThree
3. instanciate U4ScintThreeExtra which prepares prop_three from the same material fold - save to $FOLD/U4ScintThreeExtra
4. instanciate QScintThree from icdf_three/const_three/prop_three
5. instanciate QScintThree_test doing GPU wavelength generation - save to $FOLD

**/

int main()
{
    const char* material_dir = spath::Resolve("$CFBaseFromGEOM/CSGFoundry/SSim/stree/material");
    std::cout << " material_dir [" << ( material_dir ? material_dir : "-" ) << "]\n" ;
    NPFold* fold = NPFold::Load(material_dir) ;

    U4ScintThree* scint = U4ScintThree::Create(fold);
    std::cout << ( scint ? scint->desc() : "no-scint-three" ) << "\n"  ;
    if(!scint) return 0 ;
    scint->save("$FOLD/U4ScintThree");

    U4ScintThreeExtra* extra = new U4ScintThreeExtra(scint->scint, scint->name) ;
    std::cout << ( extra ? extra->desc() : "no-scint-three-extra" ) << "\n" ;
    extra->save("$FOLD/U4ScintThreeExtra");

    const NP* icdf_three  = scint->icdf_three ;
    const NP* const_three = scint->const_three ;
    const NP* prop_three  = extra->prop_three ;

    std::cout << " prop_three " << ( prop_three ? prop_three->sstr() : "-" ) << "\n" ;

    QScintThree* qs = new QScintThree( icdf_three, const_three, prop_three );
    std::cout << " qs " << ( qs ? qs->desc() : "-" ) << "\n" ;

    QScintThree_test* ts = new QScintThree_test(qs);
    ts->save("$FOLD");

    return 0 ;
}


