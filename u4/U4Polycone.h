#pragma once
/**
U4Polycone.h
===============

Initial::

    U4Polycone::desc num 4 rz 4 R_inner 2 R_outer 2 Z 3
      0 RZ     43.000    195.000      0.000
      1 RZ     43.000    195.000    -15.000
      2 RZ     55.500     70.000    -15.000
      3 RZ     55.500     70.000   -101.000

After reversal::

    U4Polycone::desc num 4 rz 4 R_inner 2 R_outer 2 Z 3
      0 RZ     55.500     70.000   -101.000
      1 RZ     55.500     70.000    -15.000
      2 RZ     43.000    195.000    -15.000
      3 RZ     43.000    195.000      0.000




       -195                 -43    :    43                  195

         0___________________0     :     0___________________0                 z = 0      (0)
         |                   |     :     |                   |
         |                   |     :     |                   |
         1___________________1     :     1___________________1                 z = -15    (1)
                     2    2        :        2    2                             z = -15    (2)   
                     |    |        :        |    |        
                     |    |        :        |    |       
                     |    |        :        |    |      
                     3____3        :        3____3                             z = -101   (3)

        -195       -70  -55.5      0       55.5  70         195


**/


#include <vector>
#include <set>
#include <algorithm>

#include "G4Polycone.hh"
#include "snd.hh"
#include "ssys.h"

struct RZ
{
    double rmin ;  
    double rmax ;  
    double z ; 

    std::string desc() const ; 
};

inline std::string RZ::desc() const
{
    std::stringstream ss ; 
    ss << "RZ" 
       << " " << std::setw(10) << std::fixed << std::setprecision(3) << rmin 
       << " " << std::setw(10) << std::fixed << std::setprecision(3) << rmax
       << " " << std::setw(10) << std::fixed << std::setprecision(3) << z 
       ; 
    std::string str = ss.str(); 
    return str ; 
}


struct U4Polycone
{
    std::string desc() const ; 
    static int  Convert( const G4Polycone* polycone ); 
    static void GetMinMax( double& mn, double& mx, const std::set<double>& vv ); 
    U4Polycone(const G4Polycone* poly); 
    bool checkZOrder( bool z_ascending ); 
    void init(); 
    void addPrims(std::vector<int>& prims, bool outside ); 

    int level ; 
    const G4Polycone* polycone ; 
    const G4PolyconeHistorical* ph ; 

    int num ; 
    std::vector<RZ> rz ;
 
    std::set<double> R_inner ; 
    std::set<double> R_outer ;
    std::set<double> Z ;

    int num_R_inner ; 
    double R_inner_min ; 
    double R_inner_max ;

    int num_R_outer ; 
    double R_outer_min ; 
    double R_outer_max ;

    int num_Z ; 
    double Z_min ; 
    double Z_max ; 

    bool   has_inner ; 
    int    root ; 
};

inline std::string U4Polycone::desc() const 
{
    std::stringstream ss ; 
    ss << "U4Polycone::desc"
       << " level " << level 
       << " num " << num 
       << " rz " << rz.size()
       << std::endl  
       << " num_R_inner " << std::setw(3) << num_R_inner
       << " R_inner_min " << std::setw(10) << R_inner_min
       << " R_inner_max " << std::setw(10) << R_inner_max
       << std::endl  
       << " num_R_outer " << std::setw(3) << num_R_outer
       << " R_outer_min " << std::setw(10) << R_outer_min
       << " R_outer_max " << std::setw(10) << R_outer_max
       << std::endl  
       << " num_Z       " << std::setw(3) << num_Z
       << " Z_min       " << std::setw(10) << Z_min
       << " Z_max       " << std::setw(10) << Z_max
       << std::endl  
       << " has_inner " << ( has_inner ? "YES" : "NO" ) 
       << " root " << std::setw(3) << root
       << std::endl 
       ;

    for(int i=0 ; i < num ; i++) ss << std::setw(3) << i << " " << rz[i].desc() << std::endl ; 
    std::string str = ss.str(); 
    return str ; 
}


inline int U4Polycone::Convert( const G4Polycone* polycone )
{
    U4Polycone upoly(polycone) ; 
    return upoly.root ; 
}

inline void U4Polycone::GetMinMax( double& mn, double& mx, const std::set<double>& vv )
{
    mn = *vv.begin() ; 
    mx = *vv.begin() ; 
    for(std::set<double>::const_iterator it = vv.begin() ; it != vv.end() ; it++ ) 
    {
        mn = std::min( mn, *it );
        mx = std::max( mx, *it );
    }
}

inline U4Polycone::U4Polycone(const G4Polycone* polycone_ ) 
    :
    level(ssys::getenvint("U4Polycone_level",0)),
    polycone(polycone_),
    ph(polycone->GetOriginalParameters()),
    num(ph->Num_z_planes),
    num_R_inner(0),
    R_inner_min(0),
    R_inner_max(0),
    num_R_outer(0),
    R_outer_min(0),
    R_outer_max(0),
    num_Z(0),
    Z_min(0),
    Z_max(0),
    has_inner(false),
    root(-1)
{
    init(); 
    if(level > 0 ) std::cerr << "U4Polycone::U4Polycone" << std::endl << desc() << std::endl ; 
}

inline bool U4Polycone::checkZOrder( bool z_ascending )
{
    int count_z_order = 0 ;
    for( int i=1 ; i < num ; i++)
    {
        bool z_order = z_ascending ? rz[i-1].z <= rz[i].z : rz[i].z <= rz[i-1].z ;
        if(z_order) count_z_order += 1 ;
    }
    bool all_z_order = count_z_order == num - 1 ; 
    return all_z_order ; 
}



inline void U4Polycone::init()
{
    rz.resize(num); 

    for (int i=0; i < num ; i++)
    {
        double rmin = ph->Rmin[i] ; 
        double rmax = ph->Rmax[i] ; 
        double z    = ph->Z_values[i] ;        

        rz[i] = { rmin, rmax, z  } ;

        R_inner.insert(rmin); 
        R_outer.insert(rmax); 
        Z.insert(z); 
    }
   
    num_R_inner = R_inner.size(); 
    num_R_outer = R_outer.size(); 
    num_Z = Z.size(); 


    bool all_z_descending = checkZOrder(false); 
    if(all_z_descending) 
    {
        if(level > 0) std::cerr 
           << "U4Polycone::init"
           << " all_z_descending detected, reversing " 
           << std::endl 
           ; 
        std::reverse( std::begin(rz), std::end(rz) ) ; 
    } 
    bool all_z_ascending  = checkZOrder(true  ); 
    assert( all_z_ascending ); 


    GetMinMax(R_inner_min, R_inner_max, R_inner); 
    GetMinMax(R_outer_min, R_outer_max, R_outer); 
    GetMinMax(Z_min, Z_max, Z); 

    assert( Z_max > Z_min ); 
    bool no_inner = R_inner_min == 0. && R_inner_max == 0. ;
    has_inner = !no_inner ;


    double startPhi = ph->Start_angle/CLHEP::radian ;  
    double deltaPhi = ph->Opening_angle/CLHEP::radian ;
    bool has_phi_segment = startPhi > 0. || deltaPhi < 2.0*CLHEP::pi  ;
    assert( has_phi_segment == false );  

    std::vector<int> outer_prims ;
    addPrims( outer_prims, true  ); // outside:true 
    snd::ZNudgeJoints(outer_prims); 

    int outer = snd::Collection(outer_prims); 
 
    if( has_inner == false )
    {
        root = outer ; 
    }
    else
    {
        int inner = -1 ; 
        if( num_R_inner == 1 )  // cylinder inner
        {
            assert( R_inner_min == R_inner_max );
            double rmin = R_inner_min ;
            inner = snd::Cylinder(rmin, Z_min, Z_max);
        }
        else
        {
            std::vector<int> inner_prims ;
            addPrims( inner_prims, false  ); // outside:false
            snd::ZNudgeEnds(inner_prims);    // only for inner as expands  
            snd::ZNudgeJoints(inner_prims); 
            inner = snd::Collection( inner_prims ); 
        }
        assert( inner > -1 ); 
        root = snd::Boolean(CSG_DIFFERENCE, outer, inner );  
    }
}
     

void U4Polycone::addPrims(std::vector<int>& prims,  bool outside  )
{   
    // loop over pairs of planes
    for( unsigned i=1 ; i < rz.size() ; i++ )
    {
        const RZ& rz1 = rz[i-1] ;   // zplane struct rmin, rmax, z
        const RZ& rz2 = rz[i] ;
        double r1 = outside ? rz1.rmax : rz1.rmin ;
        double r2 = outside ? rz2.rmax : rz2.rmin ;
        double z1 = rz1.z ;
        double z2 = rz2.z ;

        if( z1 == z2 )
        {
            if(level > 0) std::cerr << "U4Polycone::makePrims skipping prim as z2 == z1  " << std::endl ; 
            continue ;
        }

        bool z_ascending = z2 > z1 ;
        assert(z_ascending);

        int idx = r2 == r1 ? snd::Cylinder(r2, z1, z2 ) : snd::Cone( r1, z1, r2, z2 ) ; 
        prims.push_back(idx);
    }  
}




