#pragma once
/**
U4Polycone.h
===============

Polycone Z-Nudging
-------------------

Polycone Z-nudging is simpler than the general case because:

1. no transforms
2. no need to look for coincidences, as every Z-joint is coincident
   and every subtracted inner end face is coincident with the outer that
   it is subtracted from.


Polycone example
----------------

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
               rmin      rmax          z



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

#include "sn.h"
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
    ss << "RRZ"
       << " " << std::setw(10) << std::fixed << std::setprecision(3) << rmin
       << " " << std::setw(10) << std::fixed << std::setprecision(3) << rmax
       << " " << std::setw(10) << std::fixed << std::setprecision(3) << z
       ;
    std::string str = ss.str();
    return str ;
}


struct U4Polycone
{
    static sn*  Convert( const G4Polycone* polycone, int lvid, int depth, int level );
private:

    std::string desc() const ;
    static void GetMinMax( double& mn, double& mx, const std::set<double>& vv );


    U4Polycone(const G4Polycone* poly, int lvid, int depth, int level );
    bool checkZOrder( bool z_ascending );
    void init();
    void init_RZ();
    void init_phicut();
    void init_outer();
    void init_inner();

    void collectPrims(std::vector<sn*>& prims, bool outside );


    // MEMBERS

    int lvid ;
    int depth ;
    int level ;

    bool enable_nudge ;
    const G4Polycone* polycone ;
    const G4PolyconeHistorical* ph ;

    int num ;
    std::vector<RZ> rz ;


    static constexpr const char* U4Polycone__ENABLE_PHICUT = "U4Polycone__ENABLE_PHICUT" ;
    bool   ENABLE_PHICUT ;
    double phi_start ;
    double phi_end ;
    sn*    phicut ;

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

    std::vector<sn*> outer_prims ;
    std::vector<sn*> inner_prims ;
    sn*    inner ;
    sn*    outer ;
    sn*    root ;

    const char* label ;

};


inline sn* U4Polycone::Convert( const G4Polycone* polycone, int lvid, int depth, int level )
{
    U4Polycone upoly(polycone, lvid, depth, level ) ;

    if(level > 0)
    {
       std::cerr << "U4Polycone::Convert" << std::endl ;
       std::cerr << upoly.root->render(5) ;
    }
    return upoly.root ;
}




inline std::string U4Polycone::desc() const
{
    std::stringstream ss ;
    ss << "U4Polycone::desc"
       << " lvid " << lvid
       << " depth " << depth
       << " level " << level
       << " enable_nudge " << ( enable_nudge ? "YES" : "NO " )
       << " num " << num
       << " ENABLE_PHICUT " << std::setw(10) << ( ENABLE_PHICUT ? "YES" : "NO " )
       << " phi_start " << std::setw(10) << phi_start
       << " phi_end " << std::setw(10) << phi_end
       << " phicut " << std::setw(10) << ( phicut ? "YES" : "NO " )
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
       << " root " << std::setw(3) << ( root ? root->index() : -1 )
       << " label " << ( label ? label : "-" )
       << std::endl
       ;

    for(int i=0 ; i < num ; i++) ss << std::setw(3) << i << " " << rz[i].desc() << std::endl ;
    std::string str = ss.str();
    return str ;
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

inline U4Polycone::U4Polycone(const G4Polycone* polycone_, int lvid_, int depth_, int level_ )
    :
    lvid(lvid_),
    depth(depth_),
    level(level_),
    enable_nudge(!ssys::getenvbool("U4Polycone__DISABLE_NUDGE")),
    polycone(polycone_),
    ph(polycone->GetOriginalParameters()),
    num(ph->Num_z_planes),
    ENABLE_PHICUT(ssys::getenvbool(U4Polycone__ENABLE_PHICUT)),
    phi_start(0.),
    phi_end(2.*M_PI),
    phicut(nullptr),
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
    inner(nullptr),
    outer(nullptr),
    root(nullptr),
    label("NOT-WITH-SND")
{
    init();
    if(level > 0 ) std::cerr
        << "U4Polycone::U4Polycone "
        << ( label ? label : "-" )
        << std::endl
        << desc()
        << std::endl
        ;

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


/**
U4Polycone::init
-----------------

inner and outer can be sn::Cylinder OR sn::Collection (eg sn::UnionTree)
and root can be those also OR sn::Boolean difference of them.
So the technique used to support phi range needs to be applicable
to many types of node including trees and singles.

Ideas how to do that:

1. extend s_pa from 6 to 8 elem and include phi range there,
   this has disadvantage of adding empty params to almost all s_pa
   that are only used for those with phi segment (and a subsequent
   theta segment would push from 8 to 10)

2. add s_au for auxiliary params analogous to how s_pa is related to sn,
   that is quite a lot of code change

3. HMM: use boolean intersection with a special segment (CSG_PHICUT)
   type that the phi (and in future theta perhaps) range in its param

   * THIS IS ADVANTAGEOUS FROM POINT OF VIEW OF EXPRESSING THE INFO
     WITHOUT MUCH CHANGE TO EXISTING CODE

**/


inline void U4Polycone::init()
{
    init_RZ();
    init_phicut();
    init_outer();

    sn* _root = nullptr ;
    {
        if(has_inner == false)
        {
            _root = outer ;
        }
        else
        {
            init_inner();
            assert( inner );
            _root = sn::Boolean(CSG_DIFFERENCE, outer, inner );
        }
    }


    if(phicut == nullptr)
    {
        root = _root ;
    }
    else
    {
        if(ENABLE_PHICUT)
        {
            root = sn::Boolean(CSG_INTERSECTION, phicut, _root );
        }
        else
        {
            std::cerr
               << "U4Polycone::init FATAL geometry with unsupported phicut : "
               << " enable experimental support with envvar "
               << " [" <<  U4Polycone__ENABLE_PHICUT << "]"
               << "\n"
               ;

            assert(0);
            std::raise(SIGINT);
        }
    }

}

/**
U4Polycone::init_RZ
---------------------

1. fill (RZ)rz vector and insert values into std::set
   to give number of unique rmin, rmax, z

2. if necessary reverse the rz vector to make all z ascending

3. get the min/max ranges of rmin, rmax, z and determine if there
   is an inner based on the rmin range


**/

inline void U4Polycone::init_RZ()
{
    rz.resize(num);

    for (int i=0; i < num ; i++)
    {
        RZ& rzi = rz[i] ;
        rzi.rmin = ph->Rmin[i] ;
        rzi.rmax = ph->Rmax[i] ;
        rzi.z    = ph->Z_values[i] ;

        R_inner.insert(rzi.rmin);
        R_outer.insert(rzi.rmax);
        Z.insert(rzi.z);
    }

    num_R_inner = R_inner.size();
    num_R_outer = R_outer.size();
    num_Z = Z.size();


    bool all_z_descending = checkZOrder(false);
    if(all_z_descending)
    {
        if(level > 0) std::cerr
           << "U4Polycone::init_RZ"
           << label
           << " all_z_descending detected, reversing "
           << std::endl
           ;
        std::reverse( std::begin(rz), std::end(rz) ) ;
    }
    bool all_z_ascending  = checkZOrder(true  );
    assert( all_z_ascending );
    if(!all_z_ascending) std::raise(SIGINT);

    GetMinMax(R_inner_min, R_inner_max, R_inner);
    GetMinMax(R_outer_min, R_outer_max, R_outer);
    GetMinMax(Z_min, Z_max, Z);

    assert( Z_max > Z_min );
    bool no_inner = R_inner_min == 0. && R_inner_max == 0. ;
    has_inner = !no_inner ;
}


/**
U4Polycone::init_phicut
-------------------------



           Y
     .     |      n
       .   |    .
         . | .
           +------X
            .
              .


Special cased "hemi" for simpler testing::

                             +Y
                              .
                    + + + + + . + + + + +
                    + + + + + . + + + + +
                    + + + + + . + + + + +
                    + + + + + . + + + + +
                    + + + + + . + + + + +
  phi_end = pi  -X +----------+---------+ +X phi_start = 0.
                              :
                              :
                              :
                              n [0,-1,0]
                             -Y

Normal in -Y direction corresponds to the halfspace Y>0

TODO: handle all half-space situations not just one


**/

inline void U4Polycone::init_phicut()
{
    double eps = 1e-6 ;
    double phi_delta = ph->Opening_angle/CLHEP::radian ;
    phi_start = ph->Start_angle/CLHEP::radian ;
    phi_end   = phi_start + phi_delta ;

    bool has_phicut = phi_start > 0. || phi_end < 2.0*CLHEP::pi  ;
    bool has_half = std::abs( phi_start - 0.f ) < eps  && std::abs(phi_end-CLHEP::pi) < eps ;

    if( has_phicut )
    {
        if( has_half )
        {
            phicut = sn::HalfSpace( 0., -1., 0., 0. );
        }
        else
        {
            phicut = sn::PhiCut( phi_start, phi_end );
        }
    }

    if(has_phicut) std::cerr
       << "U4Polycone::init_phicut"
       << " phi_start   " << std::setw(10) << std::fixed << std::setprecision(4) << phi_start
       << " phi_end     " << std::setw(10) << std::fixed << std::setprecision(4) << phi_end
       << " phi_delta   " << std::setw(10) << std::fixed << std::setprecision(4) << phi_delta
       << " has_phicut " << ( has_phicut ? "YES" : "NO " )
       << " has_half " << ( has_half ? "YES" : "NO " )
       << " ENABLE_PHICUT " << ( ENABLE_PHICUT ? "YES" : "NO " )
       << "\n"
       ;

}

/**
U4Polycone::init_outer
------------------------

1. populate outer_prims vector with cones and cylinders
2. when more than one outer prim invoke sn::ZNudgeOverlapJoints.
   This does not change geometry as it just changes internal joints
   between prims to avoid coincident faces (assuming sane prim sizes).
3. collect the vector of prims into a binary union tree of nodes

**/

inline void U4Polycone::init_outer()
{
    if( num_R_outer == 1 )
    {
        assert( R_outer_min == R_outer_max );
        outer = sn::Cylinder( R_outer_max, Z_min, Z_max );
    }
    else
    {
        collectPrims( outer_prims, true  ); // outside:true
        int num_outer_prim = outer_prims.size() ;

        if(level > 0) std::cerr
            << "U4Polycone::init_outer."
            << " num_outer_prim " << num_outer_prim
            << std::endl
            ;

        if(num_outer_prim > 1) sn::ZNudgeOverlapJoints(lvid, outer_prims, enable_nudge);
        outer = sn::Collection(outer_prims) ;
    }

}

/**
U4Polycone::init_inner
-----------------------

1. if there is only a single inner create a single cylinder

   * HMM : what about a single cone inner ?


**/

inline void U4Polycone::init_inner()
{
    assert( has_inner ) ;

    if(level > 0) std::cerr
       << "U4Polycone::init_inner "
       << std::endl
       ;

    if( num_R_inner == 1 )  // cylinder inner
    {
        assert( R_inner_min == R_inner_max );
        inner = sn::Cylinder(R_inner_min, Z_min, Z_max);
    }
    else
    {
        collectPrims( inner_prims, false  ); // outside:false
        int num_inner_prim = inner_prims.size() ;

        if(level > 0) std::cerr
            << "U4Polycone::init."
            << label
            << " num_inner_prim " << num_inner_prim
            << std::endl
            ;

        sn::ZNudgeExpandEnds(lvid, inner_prims, enable_nudge);    // only for inner
        if(num_inner_prim > 1) sn::ZNudgeOverlapJoints(lvid, inner_prims, enable_nudge);
        inner = sn::Collection( inner_prims );
    }
}


/**
U4Polycone::collectPrims
--------------------------

Populate prims vectors with snd indices or sn pointers
to cylinder or cone nodes created using values
from the (RZ)rz vector. For outside:true use Rmax values
otherwise use Rmin values for the inner.

**/

void U4Polycone::collectPrims(std::vector<sn*>& prims,  bool outside  )
{
    bool inner = !outside ;
    // loop over pairs of planes
    int num_rz = rz.size();

    for( int i=1 ; i < num_rz ; i++ )
    {
        const RZ& rz1 = rz[i-1] ;   // zplane struct rmin, rmax, z
        const RZ& rz2 = rz[i] ;
        double r1 = outside ? rz1.rmax : rz1.rmin ;
        double r2 = outside ? rz2.rmax : rz2.rmin ;
        double z1 = rz1.z ;
        double z2 = rz2.z ;

        if( z1 == z2 )
        {
            if(level > 0) std::cerr << "U4Polycone::collectPrims skipping prim as z2 == z1  " << std::endl ;
            continue ;
        }


        bool z_ascending = z2 > z1 ;
        assert(z_ascending);
        if(!z_ascending) std::raise(SIGINT);

        bool is_cylinder = r1 == r2 ;
        int idx = -1 ;

        if( inner && r1 == 0. && r2 == 0. )
        {
            if(level > 0) std::cerr << "U4Polycone::collectPrims skipping inner as r1 == r2 == 0.  " << std::endl ;
            continue ;
        }

        /*
        if(is_cylinder)
        {
             bool r2_nonzero = r2 > 0. ;
             if(!r2_nonzero) std::cerr
                  << "U4Polycone::collectPrims"
                  << " ERROR - CYLINDER PAIR UNEXPECTED ZERO r2 "
                  << " lvid " << lvid
                  << " r2_nonzero " << ( r2_nonzero ? "YES" : "NO ")
                  << " num_rz " << num_rz
                  << " i " << i
                  << "\n"
                  << desc()
                  << "\n"
                  ;
             assert(r2_nonzero);
        }
        */

        sn* pr = is_cylinder ? sn::Cylinder(r2, z1, z2 ) : sn::Cone( r1, z1, r2, z2 ) ;
        pr->lvid = lvid ;  // so this before setting root for debug purposes
        prims.push_back(pr);
        idx = pr->index() ;

        if( level > 0 ) std::cerr
             << "U4Polycone::collectPrims"
             << " outside " << ( outside ? "YES" : "NO " )
             << " idx " << idx
             << " is_cylinder " << ( is_cylinder ? "YES" : "NO " )
             << std::endl
             ;
    }
}


