#include "SSys.hh"
#include "SPath.hh"
#include "NP.hh"

#include "QMultiFilm.hh"
#include "scuda.h"
#include "squad.h"
#include "OPTICKS_LOG.hh"


void test_check(QMultiFilm& sc)
{
    std::cout<< sc.desc()<<std::endl;
    sc.check(); 
}

void test_mock_lookup(QMultiFilm& sc){
	
	const char* input_path = "/tmp/debug_multi_film_table/test_texture.npy";
    NP* input_arr = NP::LoadIfExists(input_path); 
	assert(input_arr->has_shape(128,256,2,4));
	
	int height = input_arr->shape[0];
	int width  = input_arr->shape[1];
	int num_item = height*width;
	int edge_item = 20;
	quad2* arr_v = (quad2*)input_arr->values<float>();

	for(int i = 0 ; i < edge_item || i > num_item - edge_item; i++){
		quad2 qd2 = arr_v[i];
		std::cout<<" pmtcat = "<< qd2.q0.i.x 
				 <<" wv_nm = " << qd2.q0.f.y
				 <<" aoi = "   << qd2.q0.f.z 

				 <<" R_s = "   << qd2.q1.f.x 
				 <<" T_s = "   << qd2.q1.f.y 
				 <<" R_p = "   << qd2.q1.f.z 
				 <<" T_p = "   << qd2.q1.f.w 
				 << std::endl; 
	}
}

void test_lookup(QMultiFilm& sc)
{
    int pmtcatDim = sc.src->shape[0];
    int resDim    = sc.src->shape[1];

    //NP* dst = sc.lookup(); 
    int create_dirs = 2 ; // 2:dirpath
    const char* fold = SPath::Resolve("$TMP/QMultiFilmTest", create_dirs) ; 
    LOG(info) << " save to " << fold ; 
    for(int pmtcatIdx = 0 ; pmtcatIdx < pmtcatDim ; pmtcatIdx++){
         for( int resIdx = 0 ; resIdx < resDim ; resIdx++){
             NP* dst = sc.lookup(pmtcatIdx,resIdx );
             std::stringstream ss;
             ss<<"pmtcat_"<<pmtcatIdx
               <<"resolution_"<<resIdx
               <<".npy";
            dst->save(fold, ss.str().c_str());
         }
    }
    sc.src->save(fold, "src.npy") ; 
}

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    const char* artpath = SSys::getenvvar("ARTPATH", "/tmp/debug_multi_film_table/multifilm.npy") ; 
    NP* art = NP::LoadIfExists(artpath) ; 

    LOG(info)
       << " ARTPATH " << artpath 
       << " art " << ( art ? art->sstr() : "-" )
       ;

    LOG_IF(error, !art ) << " FAILED to load art array " ; 
    if(!art) return 0 ;  

    QMultiFilm mf(art);   

    //test_lookup(mf); 
    //test_check(mf); 
	test_mock_lookup(mf);
    return 0 ; 
}

