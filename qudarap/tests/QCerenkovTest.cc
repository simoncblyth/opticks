
#include "OPTICKS_LOG.hh"
#include "ssys.h"
#include "spath.h"
#include "scuda.h"
#include "SSim.hh"
#include "NPFold.h"

#include "QBase.hh"
#include "QOptical.hh"
#include "QBnd.hh"
#include "QProp.hh"
#include "QCerenkov.hh"


struct QCerenkovTest
{
    SSim*      ssim ;
    const NP*  optical ;
    const NP*  bnd ;
    const NP* propcom ;

    QBase*     base ;
    QOptical*  qopt ;
    QBnd*      qbnd ;
    QProp<float>* prop ;
    QCerenkov* cerenkov ;

#ifdef QCERENKOV_ICDF_OLD
    const NP*  cerenkov_icdf ;
    const NP*  cerenkov_lookup ;
#endif
    int placeholder ;

    QCerenkovTest();
    void init();
    void save();

    static int Main();

};


QCerenkovTest::QCerenkovTest()
    :
    ssim(SSim::Load()),
    optical(ssim->get_optical()),
    bnd(ssim->get_bnd()),
    propcom(ssim->get_propcom()),
    base(new QBase),
    qopt(new QOptical(optical)),
    qbnd(new QBnd(bnd)),
    prop(propcom ? new QProp<float>(propcom) : nullptr),
    cerenkov(new QCerenkov),
#ifdef QCERENKOV_ICDF_OLD
    cerenkov_icdf(cerenkov->icdf),
    cerenkov_lookup(cerenkov->lookup()),
#endif
    placeholder(0)
{
    init();
}

void QCerenkovTest::init()
{
    LOG(info) << " optical " << ( optical ? optical->sstr() : "-" ) ;
    LOG(info) << " bnd     " << ( bnd ? bnd->sstr() : "-" ) ;
    LOG(info) << " propcom " << ( propcom ? propcom->sstr() : "-" ) ;


    base->setTreeDigest(ssim->get_tree_digest());
    LOG(info) << " base.desc " << base->desc() ;

    NP_FATAL_ASSERT(optical);
    NP_FATAL_ASSERT(bnd);
    NP_FATAL_ASSERT(propcom == nullptr);  // propcom no-longer-used ?

    LOG(info) << "qopt.desc " << qopt->desc();
    LOG(info) << "qbnd.desc " << qbnd->desc();
    LOG(info) << "prop.desc " << ( prop ? prop->desc() : "-" ) << " (no-longer-used?) " ;
    LOG(info) << "cerenkov.desc " << cerenkov->desc() ;

#ifdef QCERENKOV_ICDF_OLD
    LOG(info) << "cerenkov_lookup " << ( cerenkov_lookup ? cerenkov_lookup->sstr() : "-" ) ;
    LOG(info) << "cerenkov_icdf   " << ( cerenkov_icdf   ? cerenkov_icdf->sstr() : "-" ) ;
#endif

}

int QCerenkovTest::Main()
{
    QCerenkovTest t ;
    t.save();
    return 0 ;
}


void QCerenkovTest::save()
{
#ifdef QCERENKOV_ICDF_OLD
    NPFold* f = new NPFold ;
    f->add("cerenkov_icdf",   cerenkov_icdf );
    f->add("cerenkov_lookup", cerenkov_lookup );
    f->save("$FOLD");
#endif
}


int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);
    return QCerenkovTest::Main() ;
}

