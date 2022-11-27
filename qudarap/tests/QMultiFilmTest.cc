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

    const char* artpath = SSys::getenvvar("ARTPATH", "/tmp/debug_multi_film_table/all_table.npy") ; 
    NP* art = NP::Load(artpath) ; 

    LOG(info)
       << " ARTPATH " << artpath 
       << " art " << ( art ? art->sstr() : "-" )
       ;

    LOG_IF(error, !art ) << " FAILED to load art array " ; 
    if(!art) return 0 ;  

    QMultiFilm mf(art);   

    test_lookup(mf); 
    //test_check(mf); 

    return 0 ; 
}

