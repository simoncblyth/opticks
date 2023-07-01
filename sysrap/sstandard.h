#pragma once
/**
sstandard.h
============

Creates standardized arrays of material or surface properties 
interpolated into a standard wavelength domain. 

**/

#include <limits>
#include <array>
#include "NPFold.h"
#include "sproplist.h"
#include "sdomain.h"

struct sstandard
{
    static NP* bnd(const std::vector<int4>& bd, const std::vector<std::string>& bdname, const NP* mat, const NP* sur); 
    static void column_range(int4& mn, int4& mx,  const std::vector<int4>& bd) ; 
    static NP* mat(const std::vector<std::string>& names, const NPFold* fold ); 
    static NP* sur(const std::vector<std::string>& names, const NPFold* fold ); 
    static NP* create(const sproplist* pl,  const std::vector<std::string>& names, const NPFold* fold ); 
};

/**
sstandard::bnd
----------------


Example shapes::

    mat    (20, 2, 761, 4)
    sur    (40, 2, 761, 4)
    bnd (53, 4, 2, 761, 4)

**/

inline NP* sstandard::bnd( const std::vector<int4>& bd, const std::vector<std::string>& bdname, const NP* mat, const NP* sur )
{
    assert( mat->shape.size() == 4 ); 
    assert( sur->shape.size() == 4 ); 

    int num_mat = mat->shape[0] ; 
    int num_sur = sur->shape[0] ; 

    for(int d=1 ; d < 4 ; d++) assert( mat->shape[d] == sur->shape[d] ) ; 

    assert( mat->shape[1] == sprop::NUM_PAYLOAD_GRP ); 
    int num_wl_domain = mat->shape[2] ; 
    assert( mat->shape[3] == sprop::NUM_PAYLOAD_VAL ); 

    const double* mat_v = mat->cvalues<double>(); 
    const double* sur_v = sur->cvalues<double>(); 

    int num_bnd = bd.size() ; 
    int num_bdname = bdname.size() ; 
    assert( num_bnd == num_bdname );  

    int4 mn ; 
    int4 mx ;
    column_range(mn, mx, bd ); 
    std::cout << " sstandard::bnd mn " << mn << " mx " << mx << std::endl ;  

    assert( mx.x < num_mat ); 
    assert( mx.y < num_sur ); 
    assert( mx.z < num_sur ); 
    assert( mx.w < num_mat ); 

    int ni = num_bnd ;                // ~53                 
    int nj = sprop::NUM_MATSUR ;      //   4  (omat,osur,isur,imat)
    int nk = sprop::NUM_PAYLOAD_GRP ; //   2
    int nl = num_wl_domain ;          // 761  fine domain
    int nn = sprop::NUM_PAYLOAD_VAL ; //   4
    int np = nk*nl*nn ;               // 2*761*4  number of payload values for one mat/sur 

    NP* bnd_ = NP::Make<double>(ni, nj, nk, nl, nn ); 
    bnd_->set_names( bdname ); 
    double* bnd_v = bnd_->values<double>() ; 

    for(int i=0 ; i < ni ; i++)
    {
        std::array<int, 4> _bd = {{ bd[i].x, bd[i].y, bd[i].z, bd[i].w }} ; 
        for(int j=0 ; j < nj ; j++)
        {
            int ptr     = _bd[j] ;  // omat,osur,isur,imat index "pointer" into mat or sur arrays
            if( ptr < 0 ) continue ; 
            bool is_mat =  j == 0 || j == 3 ; 
            bool is_sur =  j == 1 || j == 2 ; 
            if(is_mat) assert( ptr < num_mat ); 
            if(is_sur) assert( ptr < num_sur ); 

            int src_index = ptr*np ; 
            int dst_index = (i*nj + j)*np ; 
            const double* src_v = is_mat ? mat_v : sur_v ; 

            for(int p=0 ; p < np ; p++) bnd_v[dst_index + p] = src_v[src_index + p] ; 
        }
    }
    return bnd_  ; 
}

inline void sstandard::column_range(int4& mn, int4& mx,  const std::vector<int4>& bd)
{
    mn.x = std::numeric_limits<int>::max() ; 
    mn.y = std::numeric_limits<int>::max() ; 
    mn.z = std::numeric_limits<int>::max() ; 
    mn.w = std::numeric_limits<int>::max() ; 

    mx.x = std::numeric_limits<int>::min() ; 
    mx.y = std::numeric_limits<int>::min() ; 
    mx.z = std::numeric_limits<int>::min() ; 
    mx.w = std::numeric_limits<int>::min() ; 

    int num = bd.size(); 
    for(int i=0 ; i < num ; i++)
    {
        const int4& b = bd[i] ; 
        if(b.x > mx.x) mx.x = b.x ; 
        if(b.y > mx.y) mx.y = b.y ; 
        if(b.z > mx.z) mx.z = b.z ; 
        if(b.w > mx.w) mx.w = b.w ;

        if(b.x < mn.x) mn.x = b.x ; 
        if(b.y < mn.y) mn.y = b.y ; 
        if(b.z < mn.z) mn.z = b.z ; 
        if(b.w < mn.w) mn.w = b.w ;
    }
}


inline NP* sstandard::mat( const std::vector<std::string>& names, const NPFold* fold )
{
    const sproplist* pl = sproplist::Material() ; 
    return create(pl, names, fold ); 
}
inline NP* sstandard::sur( const std::vector<std::string>& names, const NPFold* fold )
{
    const sproplist* pl = sproplist::Surface() ; 
    return create(pl, names, fold ); 
}

/**
sstandard::create
--------------------

This operates from the NPFold props using NP interpolation. 

Perhaps it will be easier to operate at U4Tree level using Geant4 interpolation
with::

   U4Material::MakeStandardArray
   U4Surface::MakeStandardArray
 

**/

inline NP* sstandard::create(const sproplist* pl, const std::vector<std::string>& names, const NPFold* fold )
{ 
    sdomain dom ; 
     
    int ni = names.size() ;
    int nj = sprop::NUM_PAYLOAD_GRP ; 
    int nk = dom.length ; 
    int nl = sprop::NUM_PAYLOAD_VAL ; 

    NP* sta = NP::Make<double>(ni, nj, nk, nl) ; 
    sta->set_names(names); 
    double* sta_v = sta->values<double>(); 

    std::cout << "sstandard::create sta.sstr " << sta->sstr() << std::endl ; 

    for(int i=0 ; i < ni ; i++ )               // names
    {
        const char* name = names[i].c_str() ; 
        NPFold* sub = fold->get_subfold(name) ; 

        std::cout 
            << std::setw(4) << i 
            << " : "
            << std::setw(60) << name
            << " : "
            << sub->stats()
            << std::endl 
            ;

        for(int j=0 ; j < nj ; j++)           // payload groups
        {
            for(int k=0 ; k < nk ; k++)       // wavelength 
            {
                //double wavelength_nm = dom.wavelength_nm[k] ; 
                double energy_eV = dom.energy_eV[k] ; 
                double energy = energy_eV * 1.e-6 ;  // Geant4 actual energy unit is MeV

                for(int l=0 ; l < nl ; l++)   // payload values
                {
                    const sprop* prop = pl->get(j,l) ; 
                    assert( prop ); 

                    const char* pn = prop->name ; 
                    const NP* a = sub->get(pn) ; 
                    double value = a ? a->interp( energy ) : prop->def ; 

                    int index = i*nj*nk*nl + j*nk*nl + k*nl + l ; 
                    sta_v[index] = value ; 
                }
            }
        }
    }
    return sta ; 
}


