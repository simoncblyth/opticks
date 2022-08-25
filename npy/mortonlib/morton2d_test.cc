// ./morton2d_test.sh 

#include <cstdlib>
#include <cassert>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <bitset>
#include <vector>

#include "morton2d.h"
#include "NP.hh"

const char* FOLD = getenv("FOLD"); 

void morton_interleave_demo(uint64_t x , uint64_t y, const char* msg)
{
    morton2d<uint64_t> m(x,y); 
    const uint64_t& k = m.key ; 

    uint64_t x2, y2 ;
    m.decode( x2, y2 ); 

    assert( x == x2 ); 
    assert( y == y2 ); 

    std::bitset<32> sx(x) ; 
    std::bitset<32> sy(y) ; 
    std::bitset<64> sk(k) ; 

    std::cout << "morton_interleave_demo : " << msg << std::endl ; 
    std::cout << " x " << std::setw(16) << std::hex << x << std::dec << "|" << std::setw(64) << sx << std::endl ; 
    std::cout << " y " << std::setw(16) << std::hex << y << std::dec << "|" << std::setw(64) << sy << std::endl ; 
    std::cout << " k " << std::setw(16) << std::hex << k << std::dec << "|" << std::setw(64) << sk << std::endl ; 
}

void morton_interleave_demo()
{
    morton_interleave_demo( 0x00000000, 0xffffffff, "bit-by-bit interleaving x to the more significant slot than y "); 
    morton_interleave_demo( 0xffffffff, 0x00000000, "bit-by-bit interleaving x to the more significant slot than y "); 
    morton_interleave_demo( 0x77777777, 0x77777777, "bit-by-bit interleaving x to the more significant slot than y "); 
}




struct domain2d
{
    double dx[2] ; 
    double dy[2] ; 
    uint64_t scale ; 

    domain2d( double dx0, double dx1 , double dy0, double dy1, uint64_t scale_=0xffffffff )  
    {
        dx[0] = dx0 ; 
        dx[1] = dx1 ;
        dy[0] = dy0 ; 
        dy[1] = dy1 ;
        scale = scale_ ; 
    }

    double   fx_( double x ) const { return (x - dx[0])/(dx[1]-dx[0]) ; } 
    double   fy_( double y ) const { return (y - dy[0])/(dy[1]-dy[0]) ; } 
    uint64_t ix_( double x ) const { return scale*fx_(x) ; }
    uint64_t iy_( double y ) const { return scale*fy_(y) ; }

    uint64_t key_( double x, double y )
    {
        uint64_t ix = ix_(x) ; 
        uint64_t iy = iy_(y) ; 
        morton2d<uint64_t> m2(ix,iy); 
        return m2.key ;     
    }

    void decode_( double& x, double& y, uint64_t ik )
    {
        uint64_t ix, iy ;
        morton2d<uint64_t> m2(ik); 
        m2.decode( ix, iy ); 

        double fx = double(ix)/double(scale) ; 
        double fy = double(iy)/double(scale) ; 
        x = dx[0] + fx*(dx[1]-dx[0]) ; 
        y = dy[0] + fy*(dy[1]-dy[0]) ; 
    }


    static void coarsen( std::vector<uint64_t>& uu , const std::vector<uint64_t>& kk , uint64_t mask ); 
    void get_circle( std::vector<uint64_t>& kk, double radius ); 
    NP* make_array( const std::vector<uint64_t>& uu ); 

}; 





/**
domain2d::coarsen
-------------------

Attempt to coarsen coordinates by masking the least significant bits of morton code

HMM: have to mask almost all 64 bits to see reduction in counts 
Presumably this is  because the input points are in a grid, they are 
not on top of each other.  

**/

void domain2d::coarsen( std::vector<uint64_t>& uu , const std::vector<uint64_t>& kk , uint64_t mask )
{
    uu.resize(0); 
    for(unsigned i=0 ; i < kk.size() ; i++)
    {
        const uint64_t& ik0 = kk[i]  ;  
        uint64_t ik = ik0 & mask ;  

        //std::cout << "ik0" << std::hex << ik0 << std::dec << std::endl ; 
        //std::cout << "ik " << std::hex << ik  << std::dec << std::endl ; 

        if( std::find( uu.begin(), uu.end(), ik ) == uu.end() ) uu.push_back(ik) ; 
    }
}

void domain2d::get_circle( std::vector<uint64_t>& kk, double radius )
{
    for(double x=dx[0] ; x < dx[1] ; x += (dx[1]-dx[0])/200. )
    for(double y=dy[0] ; y < dy[1] ; y += (dy[1]-dy[0])/200. )
    {
        double dist = sqrtf(x*x+y*y) - radius ;    
        bool circle = std::abs(dist) < 1.1f ; 
        if(!circle) continue ; 

        uint64_t k = key_(x,y) ;  
        kk.push_back( k ); 
    }
}

NP* domain2d::make_array( const std::vector<uint64_t>& uu )
{
    NP* a = NP::Make<float>( uu.size(), 2 ) ; 
    float* aa = a->values<float>();  
    for(unsigned i=0 ; i < uu.size() ; i++) 
    { 
        const uint64_t& ik = uu[i] ;
        double x, y ; 
        decode_(x, y, ik); 
        aa[2*i+0] = x ; 
        aa[2*i+1] = y ; 
    }
    return a ; 
}









void get_morton_indices_1( std::vector<uint64_t>& kk, const double dx[2], const double dy[2], uint64_t scale, bool dump )
{
    double radius = 50.f ; 

    for(double x=dx[0] ; x < dx[1] ; x += 1.f )
    for(double y=dy[0] ; y < dy[1] ; y += 1.f )
    {
        double dist = sqrtf(x*x+y*y) - radius ;    
        bool circle = std::abs(dist) < 1.1f ; 
        if(!circle) continue ; 

        double fx = (x - dx[0])/(dx[1]-dx[0]) ; // scaled into 0->1 
        double fy = (y - dy[0])/(dy[1]-dy[0]) ; 

        uint64_t ix = scale * fx ;   // use 32 bit range 
        uint64_t iy = scale * fy ; 

        morton2d<uint64_t> m(ix,iy); 
        const uint64_t& k = m.key ; 
        kk.push_back( k ); 

        uint64_t ix2, iy2 ;
        m.decode( ix2, iy2 ); 
        assert( ix == ix2 ); 
        assert( iy == iy2 ); 

        if(dump) std::cout 
              << " x " << x 
              << " y " << y 
              << " fx " << fx 
              << " fy " << fy 
              << " ix " << ix 
              << " iy " << iy 
              << " dist " << dist 
              << std::endl 
              ;  
    }
}


void morton_circle_demo_2()
{
    domain2d dom( -100, 100, -100, 100 ); 
    std::vector<uint64_t> kk ;

    dom.get_circle(kk, 50.f ); 

    std::vector<uint64_t> uu ;
    uint64_t mask = ~0xfffffff ;   
    domain2d::coarsen(uu, kk, mask );  

    NP* a = dom.make_array( uu ); 

    a->save(FOLD, "morton_circle_demo.npy"); 
    std::cout << " save " << FOLD << " " << a->sstr() << std::endl ; 
}


void morton_circle_demo_1()
{
    double dx[2] = { -100. , 100. } ; 
    double dy[2] = { -100. , 100. } ; 
    uint64_t scale = 0xffffffff ;  

    std::vector<uint64_t> kk ;
    get_morton_indices_1( kk, dx, dy, scale, false ); 

    std::vector<uint64_t> uu ;
    //uint64_t mask = ~0 ; 
    uint64_t mask = ~0xfffffff ;   
    domain2d::coarsen(uu, kk, mask );  

    std::cout << "kk.size " << kk.size() << std::endl ; 
    std::cout << "uu.size " << uu.size() << std::endl ; 

    bool dump = false ; 

    NP* a = NP::Make<float>( uu.size(), 2 ) ; 
    float* aa = a->values<float>();  

    for(unsigned i=0 ; i < uu.size() ; i++) 
    { 
        const uint64_t& ik = uu[i] ;
        uint64_t ix, iy ;
        morton2d<uint64_t> m(ik); 
        m.decode( ix, iy ); 

        double fx = double(ix)/double(scale) ; 
        double fy = double(iy)/double(scale) ; 
        double x = dx[0] + fx*(dx[1]-dx[0]) ; 
        double y = dy[0] + fy*(dy[1]-dy[0]) ; 

        aa[2*i+0] = x ; 
        aa[2*i+1] = y ; 

        if(dump) std::cout 
            << " ik  "<< std::hex << std::setw(16) << ik << " | " << std::bitset<64>(ik) << std::endl 
            << " ix " << std::hex << std::setw(16) << ix << " | " << std::bitset<64>(ix) << std::endl 
            << " iy " << std::hex << std::setw(16) << iy << " | " << std::bitset<64>(iy) << std::endl 
            << " fx " << fx << std::endl 
            << " fy " << fy << std::endl 
            << " x " << x << std::endl 
            << " y " << y << std::endl 
            << std::endl 
            ; 
    }

    a->save(FOLD, "morton_circle_demo.npy"); 
    std::cout << " save " << FOLD << " " << a->sstr() << std::endl ; 

}










int main(int argc, char** argv)
{
    //morton_interleave_demo(); 
    //morton_circle_demo_1();  
    morton_circle_demo_2();  

    return 0 ; 
}
