
#include <iostream>
#include <iomanip>
#include <cstring>



#include "Util.h"
#include <glm/glm.hpp>
#include "glm/gtc/matrix_transform.hpp"

#include "NP.hh"
#include "Geo.h"
#include "IAS_Builder.h"
#include "GAS_Builder.h"

Geo* Geo::fGeo = NULL ; 

Geo::Geo(const char* spec_, const char* geometry_) 
   :
   spec(strdup(spec_)),  
   geometry(strdup(geometry_))
{
   fGeo = this ; 
   init();
   dumpOffsetBI(); 
}

void Geo::init()
{
    float tminf(0.1) ; 
    float tmaxf(10000.f) ; 

    if(strcmp(geometry, "sphere_containing_grid_of_two_radii_spheres_compound") == 0)
    {
        init_sphere_containing_grid_of_two_radii_spheres(tminf, tmaxf, true );
    }
    else if(strcmp(geometry, "sphere_containing_grid_of_two_radii_spheres") == 0)
    {
        init_sphere_containing_grid_of_two_radii_spheres(tminf, tmaxf, false);
    }
    else if(strcmp(geometry, "sphere") == 0 )
    {
        init_sphere(tminf, tmaxf);
    }
    else if(strcmp(geometry, "sphere_two") == 0 )
    {
        init_sphere_two(tminf, tmaxf);
    }
    else
    {
        assert(0); 
    }

    float top_extent = getTopExtent(); 
    tmin = top_extent*tminf ; 
    tmax = top_extent*tmaxf ; 
    std::cout 
        << "Geo::init" 
        << " top_extent " << top_extent  
        << " tminf " << tminf 
        << " tmin " << tmin 
        << " tmaxf " << tmaxf 
        << " tmax " << tmax 
        << std::endl 
        ; 

    float e_tminf = Util::GetEValue<float>("TMIN", -1.0) ; 
    if(e_tminf > 0.f )
    {
        tmin = top_extent*e_tminf ; 
        std::cout << "Geo::init e_tminf TMIN " << e_tminf << " override tmin " << tmin << std::endl ; 
    }
    
    float e_tmaxf = Util::GetEValue<float>("TMAX", -1.0) ; 
    if(e_tmaxf > 0.f )
    {
        tmax = top_extent*e_tmaxf ; 
        std::cout << "Geo::init e_tmaxf TMAX " << e_tmaxf << " override tmax " << tmax << std::endl ; 
    }
}

void Geo::init_sphere_containing_grid_of_two_radii_spheres(float& tminf, float& tmaxf, bool compound )
{
    std::cout << "Geo::init_sphere_containing_grid_of_two_radii_spheres " << spec << " " << ( compound ? "COMPOUND" : "no-compound" ) << std::endl ; 

    std::string gridspec = Util::GetEValue<std::string>("GRIDSPEC","-10:11,2,-10:11:2,-10:11,2") ; 
    std::array<int,9> grid ; 
    Util::ParseGridSpec(grid, gridspec.c_str());     

    int mn(0); 
    int mx(0); 
    Util::GridMinMax(grid, mn, mx); 
    int grid_extent = std::max( std::abs(mn), std::abs(mx) );  // half side
    float big_radius = float(grid_extent)*sqrtf(3.f)/2.f ;
    std::cout << "grid_extent " << grid_extent << " big_radius " << big_radius << std::endl ; 

    if(compound)
    {
        makeGAS(0.7f, 0.35f); 
        makeGAS(1.0f, 0.5f); 
    }
    else
    {
        makeGAS(0.7f); 
        makeGAS(1.0f); 
    }
    makeGAS(big_radius); 


    std::vector<unsigned> grid_modulo ;
    std::vector<unsigned> gas_single ;
    Util::GetEVector(grid_modulo, "MODULO", "0,1" ); 
    Util::GetEVector(gas_single, "SINGLE", "2" ); 
    std::cout << "MODULO " << Util::Present(grid_modulo) << std::endl ; 
    std::cout << "SINGLE " << Util::Present(gas_single) << std::endl ; 

    makeIAS_Grid(grid, grid_modulo, gas_single ); 

    setTopExtent(big_radius); 
    setTop(spec); 

    tminf = 0.75f ; 
    tmaxf = 10000.f ; 
}


void Geo::init_sphere(float& tminf, float& tmaxf)
{
    std::cout << "Geo::init_sphere" << std::endl ; 
    makeGAS(100.f); 
    setTop("g0"); 
    setTopExtent(100.f); 

    tminf = 1.60f ;   //  hmm depends on viewpoint, aiming to cut into the sphere with the tmin
    tmaxf = 10000.f ; 
}
void Geo::init_sphere_two(float& tminf, float& tmaxf)
{
    std::cout << "Geo::init_sphere_two" << std::endl ; 
    std::vector<float> extents = {100.f, 90.f, 80.f, 70.f, 60.f, 50.f   } ; 
    makeGAS(extents); 
    setTop("g0"); 
    setTopExtent(100.f); 

    tminf = 1.50f ;   //  hmm depends on viewpoint, aiming to cut into the sphere with the tmin
    tmaxf = 10000.f ; 
}

Geo* Geo::Get()
{
    return fGeo ; 
}

AS* Geo::getTop() const 
{
    return top ; 
}


/**
would need to traverse the ias and transform the bbox of all the
bi that it references in order to automate this
**/

void Geo::setTopExtent(float top_extent_)
{
    top_extent = top_extent_ ; 
}
float Geo::getTopExtent() const 
{
    return top_extent ; 
}

void Geo::setTop(const char* spec)
{
    AS* a = getAS(spec); 
    setTop(a); 
}

AS* Geo::getAS(const char* spec) const 
{
   assert( strlen(spec) > 1 ); 
   char c = spec[0]; 
   assert( c == 'i' || c == 'g' ); 
   int idx = atoi( spec + 1 ); 

   std::cout << "Geo::getAS " << spec << " c " << c << " idx " << idx << std::endl ; 

   AS* a = nullptr ; 
   if( c == 'i' )
   {
       const IAS& ias = getIAS(idx); 
       a = (AS*)&ias ; 
   } 
   else if( c == 'g' )
   {
       const GAS& gas = getGAS(idx); 
       a = (AS*)&gas ; 
   }

   if(a)
   {
       std::cout << "Geo::getAS " << spec << std::endl ; 
   } 
   return a ; 
}

void Geo::setTop(AS* top_)
{
    top = top_ ; 
}

/**
Geo::makeGAS
---------------

GAS can hold multiple bbox, but for now use just use one each 
with symmetric extent about the origin.

**/
void Geo::makeGAS(float extent)
{
    std::vector<float> extents ;
    extents.push_back(extent);  
    makeGAS(extents); 
}
void Geo::makeGAS(float extent0, float extent1)
{
    std::vector<float> extents ;
    extents.push_back(extent0);  
    extents.push_back(extent1);  
    makeGAS(extents); 
}
void Geo::makeGAS(const std::vector<float>& extents)
{
    std::cout << "Geo::makeGAS extents.size() " << extents.size() << " : " ; 
    for(unsigned i=0 ; i < extents.size() ; i++) std::cout << extents[i] << " " ; 
    std::cout << std::endl ; 

    std::vector<float> bb ; 

    // fudge enlarges bbox compared to expectation for the geometry 
    float fudge = Util::GetEValue("FUDGE", 1.0f)  ; 
    std::cout << "Geo::makeGAS fudge " << fudge << "    fudged extents : " ; 

    float extent0 = extents[0]*fudge ; 
    for(unsigned i=0 ; i < extents.size() ; i++)
    {
        float extent = extents[i]*fudge ;
        assert( extent <= extent0 );  

        std::cout << extent << " " ; 
 
        bb.push_back(-extent); 
        bb.push_back(-extent); 
        bb.push_back(-extent); 
        bb.push_back(+extent); 
        bb.push_back(+extent); 
        bb.push_back(+extent); 
    }
    std::cout << std::endl ; 


    GAS gas = {} ; 
    GAS_Builder::Build(gas, bb); 

    addGAS(gas); 
}

void Geo::addGAS(const GAS& gas)
{
    vgas.push_back(gas); 
    unsigned num_bi = gas.bis.size() ;
    nbis.push_back(num_bi); 
}

unsigned Geo::getOffsetBI(unsigned gas_idx) const 
{
    assert( gas_idx < nbis.size()); 

    unsigned offset = 0 ; 
    for(unsigned i=0 ; i < nbis.size() ; i++) 
    {
        if( i == gas_idx ) break ; 
        offset += nbis[i]; 
    }
    return offset ;     
} 


/**
Geo::getOffsetBI
------------------

Example::

    GAS_0            --> 0 
        BI_0 
        BI_1
    GAS_1            --> 2 
        BI_0 
        BI_1
    GAS_2            --> 4 
        BI_0 

Simply by keeping count of build inputs (BI) used for each GAS.

**/





unsigned Geo::getNumGAS() const 
{
    return vgas.size() ; 
}
unsigned Geo::getNumIAS() const 
{
    return vias.size() ; 
}
unsigned Geo::getNumBI() const // total 
{
    unsigned tot = 0 ; 
    for(unsigned i=0 ; i < nbis.size() ; i++) tot += nbis[i] ; 
    return tot ; 
}
unsigned Geo::getNumBI(unsigned gas_idx) const 
{
    assert( gas_idx < nbis.size()); 
    return nbis[gas_idx] ; 
}
void Geo::dumpOffsetBI() const 
{
    unsigned num_gas = getNumGAS(); 
    std::cout << "Geo::dumpOffsetBI  num_gas " << num_gas << std::endl ; 
    for(unsigned gas_idx=0 ; gas_idx < num_gas ; gas_idx++)
    {
        unsigned num_bi = getNumBI(gas_idx); 
        unsigned offset_bi = getOffsetBI(gas_idx); 
        std::cout 
            << " gas_idx " << std::setw(6) << gas_idx  
            << " num_bi " << std::setw(6) << num_bi
            << " offset_bi " << std::setw(6) << offset_bi
            << std::endl
            ;
    }
}

const GAS& Geo::getGAS(int gas_idx_) const
{
    unsigned gas_idx = gas_idx_ < 0 ? vgas.size() + gas_idx_ : gas_idx_ ;   // -ve counts from end
    assert( gas_idx < vgas.size() ); 
    return vgas[gas_idx] ; 
}
const IAS& Geo::getIAS(int ias_idx_) const
{
    unsigned ias_idx = ias_idx_ < 0 ? vias.size() + ias_idx_ : ias_idx_ ;  
    assert( ias_idx < vias.size() ); 
    return vias[ias_idx] ; 
}

float unsigned_as_float( unsigned u ) 
{
    union { unsigned u; int i; float f; } uif ;   
    uif.u = u  ;   
    return uif.f ; 
}

/**
Geo::makeIAS
-------------


gas_modulo
    vector of gas_idx which are modulo cycled in a 3d grid array  

gas_single
    vector of gas_idx which are singly included into the IAS 
    with an identity transform


Create vector of transfoms and creat IAS from that.
Currently a 3D grid of translate transforms with all available GAS repeated modulo

**/




void Geo::makeIAS_Grid(std::array<int,9>& grid, const std::vector<unsigned>& gas_modulo, const std::vector<unsigned>& gas_single )
{
    IAS ias = {}; 

    unsigned num_gas_modulo = gas_modulo.size() ; 
    unsigned num_gas_single = gas_single.size() ; 
    unsigned num_gas = getNumGAS(); 

    std::cout 
        << "Geo::makeIAS"
        << " num_gas_modulo " << num_gas_modulo
        << " num_gas_single " << num_gas_single
        << " num_gas " << num_gas
        << std::endl
        ;

    // check the input grid_idx are valid 
    for(unsigned i=0 ; i < num_gas_modulo ; i++ ) assert(gas_modulo[i] < num_gas) ; 
    for(unsigned i=0 ; i < num_gas_single ; i++ ) assert(gas_single[i] < num_gas) ; 


    // TODO: make an identity bitfield combining transform_idx and gas_idx 

    for(int i=0 ; i < int(num_gas_single) ; i++)
    {
        unsigned transform_idx = ias.trs.size() ;  // 0-based instance index within the IAS
        unsigned gas_idx = gas_single[i] ; 

        glm::mat4 tr(1.f) ;  // identity transform for the large sphere 
        tr[0][3] = unsigned_as_float(transform_idx); 
        tr[1][3] = unsigned_as_float(gas_idx) ;
        tr[2][3] = unsigned_as_float(0) ;   
        tr[3][3] = unsigned_as_float(0) ;   

        ias.trs.push_back(tr); 
    }

    for(int i=grid[0] ; i < grid[1] ; i+=grid[2] ){
    for(int j=grid[3] ; j < grid[4] ; j+=grid[5] ){
    for(int k=grid[6] ; k < grid[7] ; k+=grid[8] ){

        glm::vec3 tlat(i*1.f,j*1.f,k*1.f) ;  // grid translation 
        glm::mat4 tr(1.f) ;
        tr = glm::translate(tr, tlat );

        unsigned transform_idx = ias.trs.size();   // 0-based instance index within the IAS
        unsigned gas_modulo_idx = transform_idx % num_gas_modulo ; 
        unsigned gas_idx = gas_modulo[gas_modulo_idx] ; 

        tr[0][3] = unsigned_as_float(transform_idx); 
        tr[1][3] = unsigned_as_float(gas_idx) ;
        tr[2][3] = unsigned_as_float(0) ;   
        tr[3][3] = unsigned_as_float(0) ;   

        ias.trs.push_back(tr); 
    }
    }
    }

    IAS_Builder::Build(ias); 

    vias.push_back(ias); 
}

void Geo::writeIAS(unsigned ias_idx, const char* dir) const 
{
    const IAS& ias = getIAS(ias_idx); 

    int ni = ias.trs.size() ; 
    int nj = 4 ;
    int nk = 4 ;  

    std::cout 
        << "Geo::writeIAS"
        << " ni  " << ni
        << " nj  " << nj
        << " nk  " << nk
        << " dir " << dir
        << std::endl 
        ;

    std::stringstream ss ; 
    ss << "ias_" << ias_idx <<  ".npy" ;
    std::string s = ss.str(); 

    NP::Write(dir, s.c_str(), (float*)ias.trs.data(), ni, nj, nk ); 
}

// workaround as NP.hh not yet header-only enabled so need to include NP.hh only once 
void Geo::WriteNP( const char* dir, const char* name, float* data, int ni, int nj, int nk ) // static
{
    NP::Write(dir, name, data, ni, nj, nk ); 
}



void Geo::writeGAS(unsigned gas_idx, const char* dir) const 
{
    const GAS& gas = getGAS(gas_idx); 
    unsigned num_bi = gas.bis.size() ;

    int ni = num_bi ; 
    int nj = 2 ; 
    int nk = 3 ; 

    std::stringstream ss ; 
    ss << "gas_" << gas_idx <<  ".npy" ;
    std::string s = ss.str(); 

    std::vector<float> bb ; 
    for(unsigned i=0 ; i < num_bi ; i++)
    {
        const BI& bi = gas.bis[i]; 
        for(int j=0 ; j < 6 ; j++) bb.push_back( *(bi.aabb+j) ); 
    }

    NP::Write(dir, s.c_str(), (float*)bb.data(), ni, nj, nk ); 
}

void Geo::write(const char* dir) const 
{
    std::cout << "Geo::write " << dir << std::endl ;  

    unsigned num_gas = getNumGAS(); 
    for(unsigned i=0 ; i < num_gas ; i++) writeGAS(i, dir) ; 

    unsigned num_ias = getNumIAS(); 
    for(unsigned i=0 ; i < num_ias ; i++) writeIAS(i, dir) ; 
}

