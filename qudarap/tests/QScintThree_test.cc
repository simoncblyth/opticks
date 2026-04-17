/**
QScintThree_test.sh
====================

~/o/qudarap/tests/QScintThree_test.sh

**/

#include "spath.h"
#include "ssys.h"
#include "U4ScintThree.h"
#include "QScintThree.h"

struct QScintThree_test
{
    size_t             num_wlsamp ;
    const QScintThree* qs ;
    const NP*          wl ;

    QScintThree_test(const QScintThree* qs);
    void save(const char* base) const ;
};

inline QScintThree_test::QScintThree_test(const QScintThree* qs_)
    :
    num_wlsamp(ssys::getenvsizespec("Q4ScintThree__num_wlsamp", "M100")),
    qs(qs_),
    wl(qs ? qs->wavelength_hd20(3, num_wlsamp) : nullptr)
{
}

inline void QScintThree_test::save(const char* base) const
{
    NPFold* f = new NPFold ;
    f->add("wl", wl);
    f->save(base);
}

int main()
{
    const char* material_dir = spath::Resolve("$CFBaseFromGEOM/CSGFoundry/SSim/stree/material");
    std::cout << " material_dir [" << ( material_dir ? material_dir : "-" ) << "]\n" ;
    NPFold* fold = NPFold::Load(material_dir) ;

    U4ScintThree* scint = U4ScintThree::Create(fold);
    std::cout << ( scint ? scint->desc() : "no-scint3" ) << "\n"  ;
    if(!scint) return 0 ;
    scint->save("$FOLD");
    const NP* icdf_three = scint->icdf ;

    QScintThree* qs = new QScintThree( icdf_three );
    std::cout << " qs " << ( qs ? qs->desc() : "-" ) << "\n" ;

    QScintThree_test* ts = new QScintThree_test(qs);
    ts->save("$FOLD");

    // HMM: saving both U4ScintThree and QScintThree_test to the same FOLD is kinda messy,
    // but also kinda convenient

    return 0 ;
}


