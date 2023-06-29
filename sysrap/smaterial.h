#pragma once

#include "NPFold.h"
#include "sprop.h"

struct smaterial
{
    enum { NUM_PAYLOAD_CAT = 2, NUM_PAYLOAD_VAL = 4 } ; 
    static NP* create( const std::vector<std::string>& mtname, const NPFold* material ); 
};

NP* smaterial::create( const std::vector<std::string>& mtname, const NPFold* material )
{ 
    //NP* wl = NP::Linspace<double>( 80., 800., 800-80+1 ) ; // HMM: get from sdomain.h ?
    NP* wl = NP::Linspace<double>( 80., 800., 10 ) ; // HMM: get from sdomain.h ?
    const double* wl_v = wl->cvalues<double>() ; 

    sprop_Material pm ; 
     
    int ni = mtname.size() ;
    int nj = sprop_Material::NUM_PAYLOAD_GRP ; 
    int nk = wl->shape[0] ; 
    int nl = sprop_Material::NUM_PAYLOAD_VAL ; 

    NP* mat = NP::Make<double>(ni, nj, nk, nl) ; 
    mat->set_names(mtname); 
    double* mat_v = mat->values<double>(); 

    std::cout << "smaterial::create mat.sstr " << mat->sstr() << std::endl ; 

    for(int i=0 ; i < ni ; i++ )               // material names
    {
        const char* matname = mtname[i].c_str() ; 
        NPFold* matfold = material->get_subfold(matname) ; 

        std::cout 
            << std::setw(4) << i 
            << " : "
            << std::setw(25) << matname
            << " : "
            << matfold->stats()
            << std::endl 
            ;

        for(int j=0 ; j < nj ; j++)           // payload groups
        {
            for(int k=0 ; k < nk ; k++)       // wavelength 
            {
                double wavelength = wl_v[k] ; 
                for(int l=0 ; l < nl ; l++)   // payload values
                {
                    const sprop* prop = pm.get(j,l) ; 
                    const char* name = prop->name ; 
                    const NP* a = matfold->get(name) ; 
                    std::cout << " name " << name << " a " << ( a ? a->sstr() : "-" )  << std::endl ;  

                    int index = i*nj*nk*nl + j*nk*nl + k*nl + l ; 
                    mat_v[index] = a ? a->interp( wavelength ) : 0. ; 
                    // hmm is it wavelength domain ?
                }
            }
        }
    }
    return mat ; 
}

