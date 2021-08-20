#include "scuda.h"
#include "squad.h"

#include "QEvent.hh"
#include "QBuf.hh"
#include "QSeed.hh"

template struct QBuf<quad6> ; 


QEvent* QEvent::MakeFake()
{
    std::vector<int> photon_counts_per_genstep = { 3, 5, 2, 0, 1, 3, 4, 2, 4 };  
    QBuf<quad6> gs = QSeed::UploadFakeGensteps(photon_counts_per_genstep) ;

    QEvent* qe = new QEvent ; 
    qe->setGensteps(gs); 
    return qe ; 
}

void QEvent::setGensteps(QBuf<quad6> gs_ )
{
    gs = gs_ ; 
    se = QSeed::CreatePhotonSeeds(gs);
}


  

