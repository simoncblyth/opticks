#include "SSys.hh"
#include "SPath.hh"
#include "NP.hh"

#include "QMultiFilm.hh"
#include "scuda.h"
#include "OPTICKS_LOG.hh"


void test_check(QMultiFilm& sc)
{
    std::cout<< sc.desc()<<std::endl;
    sc.check(); 
}

void test_lookup(QMultiFilm& sc)
{

    int pmtcatDim = sc.src->shape[0];
    int bndDim    = sc.src->shape[1];
    int resDim    = sc.src->shape[2];

    //NP* dst = sc.lookup(); 
    int create_dirs = 2 ; // 2:dirpath
    const char* fold = SPath::Resolve("$TMP/QMultiFilmTest", create_dirs) ; 
    LOG(info) << " save to " << fold ; 
    for(int pmtcatIdx = 0 ; pmtcatIdx < pmtcatDim ; pmtcatIdx++){
        for(int bndIdx = 0 ; bndIdx < bndDim ; bndIdx++){
             for( int resIdx = 0 ; resIdx < resDim ; resIdx++){
                 
                 NP* dst = sc.lookup(pmtcatIdx , bndIdx , resIdx );
                 std::stringstream ss;
                 ss<<"pmtcat_"<<pmtcatIdx
                   <<"bnd_"<<bndIdx
                   <<"resolution_"<<resIdx
                   <<".npy";
                 
                dst->save(fold, ss.str().c_str());
             }
        }
    }
    sc.src->save(fold, "src.npy") ; 
}

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    //Opticks ok(argc, argv); 
    //ok.configure(); 
/*
#ifdef OLD_WAY
    GScintillatorLib* slib = GScintillatorLib::load(&ok);
    slib->dump();
    NP* icdf = slib->getBuf(); 
#else
    const char* cfbase = ok.getFoundryBase("CFBASE") ; 
    NP* icdf = NP::Load(cfbase, "CSGFoundry", "icdf.npy"); // HMM: this needs a more destinctive name/location  
#endif
    //icdf->dump(); 

    unsigned hd_factor = 0u ; 
*/
    NP* icdf = NP::Load("/tmp/debug_multi_film_table/","all_table.npy");

    QMultiFilm sc(icdf);     // uploads reemission texture  

    test_lookup(sc); 
    //test_check(sc); 

    return 0 ; 
}

