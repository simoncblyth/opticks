/**
sphoton_test.cc
=================

::

     ~/o/sysrap/tests/sphoton_test.sh
     TEST=set_lpos ~/o/sysrap/tests/sphoton_test.sh

     OFFSET=100,100,100 ~/o/sysrap/tests/sphoton_test.sh
     OFFSET=1000,1000,1000 ~/o/sysrap/tests/sphoton_test.sh


**/

#include <iostream>
#include <array>
#include <bitset>

#include "scuda.h"
#include "squad.h"
#include "sphoton.h"
#include "ssys.h"
#include "NPX.h"
#include "NPFold.h"

#include "OpticksPhoton.h"

struct sphoton_test
{
    static std::string dump_(const char* label, unsigned mask);
    static void dump( const char* label, unsigned mask);

    static int qphoton_();
    static int cast();
    static int ephoton_();
    static int make_ephoton_array();
    static int sphoton_selector_();
    static int set_flag();
    static int addto_flagmask();
    static int change_flagmask();
    static int digest();
    static int sphotond_();
    static int Get();
    static int dGet();
    static int set_polarization();
    static int dot_pol_cross_mom_nrm();
    static int make_record_array();
    static int ChangeTimeInsitu();

    static int index();
    static int demoarray();
    static int set_lpos();

    static NP* MockupPhotonsWithContiguousIIndex(const std::vector<size_t>& starts);
    static int GetIndexBeyondCursorContiguousIIndex();
    static int GetContiguousIIndexStartIndices();

    static int main();
};


std::string sphoton_test::dump_(const char* label, unsigned mask)
{
    std::bitset<32> bs(mask);
    std::stringstream ss ;
    ss << std::setw(25) << label << " " << bs << " " << std::setw(2) << bs.count() ;
    std::string s = ss.str();
    return s ;
}
void sphoton_test::dump( const char* label, unsigned mask)
{
    std::cout << dump_(label, mask) << std::endl ;
}

int sphoton_test::qphoton_()
{
#ifdef WITH_QPHOTON
    qphoton qp ;
    qp.q.zero();
    std::cout << qp.q.desc() << std::endl ;
#else
    assert(0);
#endif
    return 0;
}

int sphoton_test::cast()
{
    sphoton p ;
    quad4& q = (quad4&)p ;
    q.zero();

    p.wavelength = 501.f ;

    std::cout << q.desc() << std::endl ;
    std::cout << p.desc() << std::endl ;
    return 0;
}
int sphoton_test::ephoton_()
{
    sphoton p ;
    p.ephoton();
    std::cout << p.desc() << std::endl ;
    return 0;
}
int sphoton_test::make_ephoton_array()
{
    NP* p = sphoton::make_ephoton_array(8);
    std::cout
        << " p " << ( p ? p->sstr() : "-" )
        << std::endl
        << " p.repr " << ( p ? p->repr<float>() : "-" )
        << std::endl
        ;
    p->save("$FOLD/sphoton_test_make_ephoton_array.npy") ;
    return 0;
}

int sphoton_test::sphoton_selector_()
{
    sphoton p ;
    p.ephoton();

    unsigned hitmask = 0xdeadbeef ;
    sphoton_selector s(hitmask) ;
    assert( s(p) == false );

    p.set_flag(hitmask);
    assert( s(p) == true );
    return 0;
}

int sphoton_test::set_flag()
{
    sphoton p = {} ;
    std::cout << "set_flag.0 : " << p.descFlag() << "\n" ;

    p.set_flag( SURFACE_DETECT );
    std::cout << "set_flag.1 : " << p.descFlag() << "\n" ;

    p.set_flag( EFFICIENCY_COLLECT );
    std::cout << "set_flag.2 : " << p.descFlag() << "\n" ;

    return 0 ;
}

int sphoton_test::addto_flagmask()
{
    sphoton p = {} ;
    std::cout << "addto_flagmask.0 : " << p.descFlag() << "\n" ;

    p.addto_flagmask( SURFACE_DETECT );
    std::cout << "addto_flagmask.1 SURFACE_DETECT : " << p.descFlag() << "\n" ;

    p.addto_flagmask( EFFICIENCY_COLLECT );
    std::cout << "addto_flagmask.2 EFFICIENCY_COLLECT : " << p.descFlag() << "\n" ;

    p.addto_flagmask( EFFICIENCY_CULL );
    std::cout << "addto_flagmask.3 EFFICIENCY_CULL : " << p.descFlag() << "\n" ;



    return 0 ;
}






/**

                     zero 00000000000000000000000000000000  0
              |= CERENKOV 00000000000000000000000000000001  1
         |= SCINTILLATION 00000000000000000000000000000011  2
                  |= MISS 00000000000000000000000000000111  3
           |= BULK_ABSORB 00000000000000000000000000001111  4
           |= BULK_REEMIT 00000000000000000000000000011111  5
          |= BULK_SCATTER 00000000000000000000000000111111  6
        |= SURFACE_DETECT 00000000000000000000000001111111  7
        |= SURFACE_ABSORB 00000000000000000000000011111111  8
      |= SURFACE_DREFLECT 00000000000000000000000111111111  9
      |= SURFACE_SREFLECT 00000000000000000000001111111111 10
      |= BOUNDARY_REFLECT 00000000000000000000011111111111 11
     |= BOUNDARY_TRANSMIT 00000000000000000000111111111111 12
                 |= TORCH 00000000000000000001111111111111 13
             |= NAN_ABORT 00000000000000000011111111111111 14
       |= EFFICIENCY_CULL 00000000000000000111111111111111 15
    |= EFFICIENCY_COLLECT 00000000000000001111111111111111 16
          &= ~BULK_ABSORB 00000000000000001111111111110111 15
          &= ~BULK_REEMIT 00000000000000001111111111100111 14
          |=  BULK_REEMIT 00000000000000001111111111110111 15
          |=  BULK_ABSORB 00000000000000001111111111111111 16

**/

int sphoton_test::change_flagmask()
{
    unsigned flagmask = 0u ;         dump("zero", flagmask );
    flagmask |= CERENKOV ;           dump("|= CERENKOV", flagmask );
    flagmask |= SCINTILLATION ;      dump("|= SCINTILLATION", flagmask );
    flagmask |= MISS ;               dump("|= MISS", flagmask );
    flagmask |= BULK_ABSORB ;        dump("|= BULK_ABSORB", flagmask );
    flagmask |= BULK_REEMIT ;        dump("|= BULK_REEMIT", flagmask );
    flagmask |= BULK_SCATTER ;       dump("|= BULK_SCATTER", flagmask );
    flagmask |= SURFACE_DETECT ;     dump("|= SURFACE_DETECT", flagmask );
    flagmask |= SURFACE_ABSORB ;     dump("|= SURFACE_ABSORB", flagmask );
    flagmask |= SURFACE_DREFLECT ;   dump("|= SURFACE_DREFLECT", flagmask );
    flagmask |= SURFACE_SREFLECT ;   dump("|= SURFACE_SREFLECT", flagmask );
    flagmask |= BOUNDARY_REFLECT ;   dump("|= BOUNDARY_REFLECT", flagmask );
    flagmask |= BOUNDARY_TRANSMIT ;  dump("|= BOUNDARY_TRANSMIT", flagmask );
    flagmask |= TORCH ;              dump("|= TORCH", flagmask);
    flagmask |= NAN_ABORT ;          dump("|= NAN_ABORT", flagmask );
    flagmask |= EFFICIENCY_CULL ;    dump("|= EFFICIENCY_CULL", flagmask );
    flagmask |= EFFICIENCY_COLLECT ; dump("|= EFFICIENCY_COLLECT", flagmask) ;
    flagmask &= ~BULK_ABSORB ;       dump("&= ~BULK_ABSORB", flagmask);
    flagmask &= ~BULK_REEMIT ;       dump("&= ~BULK_REEMIT", flagmask);
    flagmask |=  BULK_REEMIT ;       dump("|=  BULK_REEMIT", flagmask);
    flagmask |=  BULK_ABSORB ;       dump("|=  BULK_ABSORB", flagmask);
    return 0;
}

int sphoton_test::digest()
{
    sphoton p ;
    p.ephoton();

    std::cout
        << " p.digest()   " << p.digest() << std::endl
        << " p.digest(16) " << p.digest(16) << std::endl
        << " p.digest(12) " << p.digest(12) << std::endl
        ;

    return 0;
}

int sphoton_test::sphotond_()
{
    sphoton  f ;
    sphotond d ;

    assert( sizeof(d) == 2*sizeof(f) );
    assert( sizeof(f) == 16*sizeof(float) );
    assert( sizeof(d) == 16*sizeof(double) );
    return 0;
}

int sphoton_test::Get()
{
    std::cout << "test_sphoton::Get" << std::endl ;
    float time_0 =  3.f ;
    float time_1 = 13.f ;

    NP* a = NP::Make<float>(2, 4, 4);
    std::array<float, 32> vv = {{
       0.f,  1.f,  2.f,  time_0,
       4.f,  5.f,  6.f,  7.f,
       8.f,  9.f, 10.f, 11.f,
      12.f, 13.f, 14.f, 15.f,

      10.f, 11.f, 12.f, time_1,
      14.f, 15.f, 16.f, 17.f,
      18.f, 19.f, 20.f, 21.f,
      22.f, 23.f, 24.f, 25.f
     }};

    memcpy( a->values<float>(), vv.data(), sizeof(float)*vv.size() );

    sphoton p0 ;
    sphoton::Get(p0, a, 0 );
    assert( p0.time == time_0 );

    sphoton p1 ;
    sphoton::Get(p1, a, 1 );
    assert( p1.time == time_1 );
    return 0;
}

int sphoton_test::dGet()
{
    std::cout << "test_sphoton::dGet" << std::endl ;

    double time_0 =  3. ;
    double time_1 = 13. ;

    NP* a = NP::Make<double>(2, 4, 4);
    std::array<double, 32> vv = {{
       0.,  1.,  2.,  time_0,
       4.,  5.,  6.,  7.,
       8.,  9., 10., 11.,
      12., 13., 14., 15.,

      10., 11., 12., time_1,
      14., 15., 16., 17.,
      18., 19., 20., 21.,
      22., 23., 24., 25.
     }};

    memcpy( a->values<double>(), vv.data(), sizeof(double)*vv.size() );

    sphotond p0 ;
    sphotond::Get(p0, a, 0 );
    assert( p0.time == time_0 );

    sphotond p1 ;
    sphotond::Get(p1, a, 1 );
    assert( p1.time == time_1 );
    return 0;
}


/**
set_polarization
-----------------

::

    .      nrm
            Z
     mom    |
        \   |
         \  |
          \ |
           \|
     -------+--------- X


* pol:transverse to mom
* trv:perpendicular to plane of incidence (-Y axis into page)


**/

int sphoton_test::set_polarization()
{
    std::cout << "sphoton_test::set_polarization" << std::endl ;

    float3 nrm = make_float3(0.f, 0.f, 1.f) ;
    float3 mom = normalize(make_float3( 1.f, 0.f, -1.f ));

    float3 trv = normalize(cross(mom, nrm )) ;   // -Y
    float3 trv1 = normalize(cross(nrm, mom )) ;  // +Y

    std::cout
        << " nrm "  << nrm << std::endl
        << " mom "  << mom << std::endl
        << " trv "  << trv << std::endl
        << " trv1 "  << trv1 << std::endl
        ;

    sphoton p ;
    p.zero();
    p.mom = mom ;

    const int N = 16 ;
    for(int i=0 ; i <= N ; i++)
    {
        float frac_twopi = float(i)/float(N)  ;
        p.set_polarization(frac_twopi) ;

        std::cout
            << p.descDir()
            << " frac_twopi " << std::fixed << std::setw(7) << std::setprecision(3) << frac_twopi
            //<< " dot(p.mom,p.pol) " << std::fixed << std::setw(7) << std::setprecision(3) << dot(p.mom,p.pol)
            << " dot(p.pol, nrm) " << std::fixed << std::setw(7) << std::setprecision(3) <<  dot(p.pol, nrm)
            << " dot(p.pol, trv) " << std::fixed << std::setw(7) << std::setprecision(3) <<  dot(p.pol, trv)
            << std::endl
            ;

        // dot(p.pol, nrm) zero where p.pol is orthogonal to the normal

    }
    return 0;
}



/**
dot_pol_cross_mom_nrm
-----------------------


    mom      nrm
       .      |
         .    |
           .  |
             .|
     ---------+----------

* mom and nrm vectors define the plane of incidence, they lie within it
* pol is transverse to mom, so there is a circle of possible directions including

  * transverse to plane of incidence (S polarized)
  * within the plane of incidence (P polarized)
  * combination of S and P polarizations by
    sweeping the pol vector around the mom vector
    whilst staying transverse to the mom vector


**/


int sphoton_test::dot_pol_cross_mom_nrm()
{
    printf("//sphoton_test::dot_pol_cross_mom_nrm \n");

    float3 nrm = normalize(make_float3(0.f, 0.f, 1.f)) ;

    //float3 mom = normalize(make_float3(1.f, 0.f, -1.f ));  // 45 degrees
    //float3 mom = normalize(make_float3(2.f, 0.f, -1.f ));  // shallower
    float3 mom = normalize(make_float3(1.f, 0.f, -2.f ));    // steeper


    float3 tra = cross( mom, nrm ) ;
    float ltr = length(tra) ;
    float mct = dot(mom,nrm) ;
    float st = sqrt( 1.f - mct*mct );
    float llmm = ltr*ltr + mct*mct ;   // tis close to 1.f

    std::stringstream ss ;
    ss
        << " nrm "  << nrm
        << " mom "  << mom
        << " tra "  << tra
        << " mct "  << mct
        << " st "  << st
        << " ltr "  << ltr
        << " llmm "  << llmm
        ;

    std::string str = ss.str();
    std::cout << str << std::endl ;


    sphoton p ;
    p.zero();
    p.mom = mom ;

    const int N = ssys::getenvint("N", 16)  ;

    NP* a = NP::Make<float>(N, 4);
    float* aa = a->values<float>();

    for(int i=0 ; i < N ; i++)
    {
        float frac_twopi = float(i)/float(N)  ;
        p.set_polarization(frac_twopi) ;
        float check_pol_mom_transverse = dot(p.mom,p.pol) ;
        assert( std::abs(check_pol_mom_transverse) < 1e-6f ) ;

        float pot = dot( p.pol, tra ) ;  // value is cos(pol-trans-angle)*sin(mom-nrm-angle)
        float pot_st = pot/st ;   // this is spol_frac which stays in range from -1. to 1.
        float pot_mct = pot/mct ; // some unholy combination that is not constrained to -1. to 1.

        std::cout
            << p.descDir()
            << " frac_twopi " << std::fixed << std::setw(7) << std::setprecision(3) << frac_twopi
            << " pot " << std::fixed << std::setw(7) << std::setprecision(3) << pot
            << " pot_mct " << std::fixed << std::setw(7) << std::setprecision(3) << pot_mct
            << " pot_st " << std::fixed << std::setw(7) << std::setprecision(3) << pot_st
            << std::endl
            ;

        aa[i*4+0] = frac_twopi ;
        aa[i*4+1] = pot ;
        aa[i*4+2] = pot_mct ;
        aa[i*4+3] = pot_st ;
    }
    a->save("$FOLD/dot_pol_cross_mom_nrm.npy");
    return 0;
}


/**
sphoton_test::make_record_array
---------------------------------

::

    In [15]: np.min(t.record[:,:,0].reshape(-1,4),axis=0)
    Out[15]: array([-9.,  0., -9.,  0.], dtype=float32)

    In [16]: np.max(t.record[:,:,0].reshape(-1,4),axis=0)
    Out[16]: array([9., 0., 9., 9.], dtype=float32)


**/



int sphoton_test::make_record_array()
{
    std::vector<float>* offset = ssys::getenvfloatvec("OFFSET","0,0,0");
    assert( offset && offset->size() == 3 );

    std::cout << "sphoton_test::make_record_array OFFSET [ " ;
    for(int i=0 ; i < 3 ; i++) std::cout
         << std::fixed << std::setw(10) << std::setprecision(4) << (*offset)[i]
         ;
    std::cout << " ]" << std::endl ;


    NP* a = NP::Make<float>( 360, 10, 4, 4 );
    int ni = a->shape[0] ;
    int nj = a->shape[1] ;
    int nk = a->shape[2] ;
    int nl = a->shape[3] ;
    float* aa = a->values<float>();

    for(int i=0 ; i < ni ; i++)
    {
        float t = 2.f*M_PIf * float(i)/360.  ;
        float ct = cos(t);
        float st = sin(t);

        for(int j=0 ; j < nj ; j++)
        {
            int idx_ij = i*nj*nk*nl + j*nk*nl ;
            sphoton p = {} ;

            // XZ ripples on pond heading outwards from origin
            p.pos.x = (*offset)[0] + ct*float(j) ;
            p.pos.y = (*offset)[1] + 0.f ;
            p.pos.z = (*offset)[2] + st*float(j) ;
            p.time = float(j) ;

            p.mom.x = ct ;
            p.mom.y = 0.f ;
            p.mom.z = st ;

            memcpy( aa + idx_ij, p.cdata(), sizeof(float)*nk*nl );
        }
    }
    a->set_meta<std::string>("rpos", "4,GL_FLOAT,GL_FALSE,64,0,false" );
    // Q:What reads this OpenGL attribute metadata ?
    // A:sysrap/SGLFW.h SGLFW_Attribute


    static const int N = 4 ;
    float mn[N] = {} ;
    float mx[N] = {} ;

    int item_stride = 4 ;
    int item_offset = 0 ;

    a->minmax2D_reshaped<N,float>(mn, mx, item_stride, item_offset );
    for(int j=0 ; j < N ; j++) std::cout
          << std::setw(2) << j
          << " mn "
          << std::setw(10) << std::fixed << std::setprecision(4) << mn[j]
          << " mx "
          << std::setw(10) << std::fixed << std::setprecision(4) << mx[j]
          << std::endl
          ;

    a->save("$FOLD/record.npy");
    return 0;
}


int sphoton_test::ChangeTimeInsitu()
{
    std::vector<sphoton> pp(3) ;
    pp[0] = {} ;
    pp[1] = {} ;
    pp[2] = {} ;

    pp[0].time = 0.1f ;
    pp[1].time = 0.2f ;
    pp[2].time = 0.3f ;

    NP* a = NPX::ArrayFromVec<float, sphoton>(pp, 4, 4);
    sphoton::ChangeTimeInsitu(a, 100.f);

    NPFold* f = new NPFold ;
    f->add("a", a );
    f->save("$FOLD/ChangeTimeInsitu");

    return 0 ;
}

int sphoton_test::index()
{
    sphoton p = {} ;
    std::vector<uint64_t> xx = {
         0x0,
         0x1,
         0xff,
         0xffff,
         0xffffff,
         0xffffffff,
         0xffffffffff,
         0xffffffffffff,
         0xffffffffffffff
     };

    std::vector<uint64_t> ii = {
         0x0,
         0x1,
         0xff,
         0xffff,
         0xffffff,
         0xffffffff,
         0xffffffffff,
         0xffffffffffff,
         0xffffffffffffff
     };


    uint64_t INDEX_MAX = 0xffffffffff ;  //  0xffffffffff/1e9 = 1099.511   1.099 trillion
    unsigned IDENTITY_MAX = 0xffffffu ;  // 0xffffff/1e6 = 16.777215    16.7 million

    for(size_t i=0 ; i < xx.size() ; i++)
    {
        uint64_t x0 = xx[i];
        unsigned i0 = ii[i]; // narrow

        p.set_index(x0);
        p.set_identity(i0);

        uint64_t x1 = p.get_index();
        unsigned i1 = p.get_identity();

        std::cout
           << std::setw(4) << i
           << " x0 "
           << std::setw(15) << std::hex << x0 << std::dec
           << " x1 "
           << std::setw(15) << std::hex << x1 << std::dec
           << " ( x0 & INDEX_MAX ) "
           << std::setw(15) << std::hex << ( x0 & INDEX_MAX ) << std::dec
           << " ( x1 & INDEX_MAX ) "
           << std::setw(15) << std::hex << ( x1 & INDEX_MAX ) << std::dec
           << " i0 "
           << std::setw(15) << std::hex << i0 << std::dec
           << " i1 "
           << std::setw(15) << std::hex << i1 << std::dec
           << " ( i0 & IDENTITY_MAX ) "
           << std::setw(15) << std::hex << ( x0 & IDENTITY_MAX ) << std::dec
           << " ( x1 & IDENTITY_MAX ) "
           << std::setw(15) << std::hex << ( x1 & IDENTITY_MAX ) << std::dec
           << "\n"
           ;

        assert( ( x0 & INDEX_MAX ) == x1 ) ;
        assert( ( i0 & IDENTITY_MAX ) == i1 ) ;
    }
    return 0 ;
}

int sphoton_test::demoarray()
{
    NP* p = sphoton::demoarray(10);
    p->save("$FOLD/demoarray.npy");
    return 0 ;
}

int sphoton_test::set_lpos()
{
     sphoton p = {};

     float eps = 1e-6 ;  // 1e-7 has one fphi deviant
     int ni = 1000 ;

     std::cout
         << "[sphoton_test::set_lpos"
         << " eps "  << std::setw(10) << std::setprecision(8) << std::fixed << eps
         << " ni " << ni
         << "\n"
         ;

     int deviant = 0 ;
     for(int i=0 ; i < ni ; i++)
     {
         //float f = float(i)/float(ni) ;
         float f = float(i+1)/float(ni+1) ;  // avoid zero and one

         float cost_0 = f;
         float fphi_0 = f;
         p.set_lpos(cost_0, fphi_0);
         float cost_1 = p.get_cost();
         float fphi_1 = p.get_fphi();

         float cost_01 = ( cost_0 - cost_1 );
         float fphi_01 = ( fphi_0 - fphi_1 ) ;

         bool select = std::abs(cost_01) > eps || std::abs(fphi_01) > eps ;
         if(select)
         {
             deviant += 1 ;
             std::cout
                 << std::setw(5) << i
                 << " cost_0 "  << std::setw(10) << std::setprecision(6) << std::fixed << cost_0
                 << " cost_1 "  << std::setw(10) << std::setprecision(6) << std::fixed << cost_1
                 << " cost_01*1e6 " << std::setw(10) << std::setprecision(6) << std::fixed << cost_01*1e6
                 << " fphi_0 "  << std::setw(10) << std::setprecision(6) << std::fixed << fphi_0
                 << " fphi_1 "  << std::setw(10) << std::setprecision(6) << std::fixed << fphi_1
                 << " fphi_01*1e6 " << std::setw(10) << std::setprecision(6) << std::fixed << fphi_01*1e6
                 << " p.pos.x " << std::setw(10) << std::setprecision(6) << std::fixed << p.pos.x
                 << " p.pos.y " << std::setw(10) << std::setprecision(6) << std::fixed << p.pos.y
                 << " p.pos.z " << std::setw(10) << std::setprecision(6) << std::fixed << p.pos.z
                 << "\n"
                 ;

         }
    }

     std::cout
         << "]sphoton_test::set_lpos deviant " << deviant << "\n" ;



    return 0 ;
}

/**
sphoton_test::MockupPhotonsWithContiguousIIndex
-------------------------------------------------

Creates an array of *ni* "starts[-1]" photons.
For the i-th photon the index of the *starts* ranges
that *i* is part of gives a *j* index that yields 
an *ii* result.

**/


NP* sphoton_test::MockupPhotonsWithContiguousIIndex(const std::vector<size_t>& starts) // static
{
    size_t nj = starts.size();
    size_t ni = starts[nj-1] ;

    NP* photons = sphoton::zeros(ni);
    sphoton* pp = (sphoton*)photons->bytes();
    for(size_t i=0 ; i < ni ; i++ )
    {
        unsigned ii = 0 ;
        for(size_t j=1 ; j < nj ; j++)
        {
            bool j_range = i >= starts[j-1] && i < starts[j] ;
            if( j_range )
            {
                ii = j*100 ;
                break ;
            }
        }
        

        pp[i].set_iindex__( ii );
    }
    return photons ;
}

int sphoton_test::GetIndexBeyondCursorContiguousIIndex()
{
    std::vector<size_t> starts = { 0, 100, 200, 400, 600, 800, 1000 };
    NP* photons = MockupPhotonsWithContiguousIIndex(starts);

    sphoton* pp = (sphoton*)photons->bytes();
    size_t num = photons->num_items();

    std::cout
        << "sphoton_test::GetIndexBeyondCursorContiguousIIndex"
        << "\n"
        ;


    std::vector<size_t> recover ;

    size_t cursor = 0 ;
    recover.push_back(cursor);
    do
    {
        cursor = sphoton::GetIndexBeyondCursorContiguousIIndex(pp, num, cursor );
        recover.push_back(cursor);
        std::cout << " cursor " << cursor << "\n" ;
    }
    while( cursor < num );
    assert( recover == starts );

    return 0 ;
}

int sphoton_test::GetContiguousIIndexStartIndices()
{
    std::vector<size_t> starts = { 0, 100, 200, 400, 600, 800, 1000 };
    NP* photons = MockupPhotonsWithContiguousIIndex(starts);
    sphoton* pp = (sphoton*)photons->bytes();
    size_t num = photons->num_items();

    std::vector<size_t> recover ;
    sphoton::GetContiguousIIndexStartIndices(recover, pp, num );
    assert( recover == starts );

    return 0 ;
}




int sphoton_test::main()
{
    const char* TEST = ssys::getenvvar("TEST","make_record_array") ;
    bool ALL = 0 == strcmp(TEST, "ALL");

    int rc = 0 ;

    if(ALL||0==strcmp(TEST, "qphoton_"))              rc += qphoton_();
    if(ALL||0==strcmp(TEST, "cast"))                  rc += cast();
    if(ALL||0==strcmp(TEST, "ephoton_"))              rc += ephoton_();
    if(ALL||0==strcmp(TEST, "make_ephoton_array"))    rc += make_ephoton_array();
    if(ALL||0==strcmp(TEST, "sphoton_selector_"))     rc += sphoton_selector_();
    if(ALL||0==strcmp(TEST, "set_flag"))              rc += set_flag();
    if(ALL||0==strcmp(TEST, "addto_flagmask"))        rc += addto_flagmask();
    if(ALL||0==strcmp(TEST, "change_flagmask"))       rc += change_flagmask();
    if(ALL||0==strcmp(TEST, "digest"))                rc += digest();
    if(ALL||0==strcmp(TEST, "sphotond_"))             rc += sphotond_();
    if(ALL||0==strcmp(TEST, "Get"))                   rc += Get();
    if(ALL||0==strcmp(TEST, "dGet"))                  rc += dGet();
    if(ALL||0==strcmp(TEST, "set_polarization"))      rc += set_polarization();
    if(ALL||0==strcmp(TEST, "dot_pol_cross_mom_nrm")) rc += dot_pol_cross_mom_nrm();
    if(ALL||0==strcmp(TEST, "make_record_array"))     rc += make_record_array();
    if(ALL||0==strcmp(TEST, "ChangeTimeInsitu"))      rc += ChangeTimeInsitu();
    if(ALL||0==strcmp(TEST, "index"))                 rc += index();
    if(ALL||0==strcmp(TEST, "demoarray"))             rc += demoarray();
    if(ALL||0==strcmp(TEST, "set_lpos"))              rc += set_lpos();

    if(ALL||0==strcmp(TEST, "GetIndexBeyondCursorContiguousIIndex")) rc += GetIndexBeyondCursorContiguousIIndex();
    if(ALL||0==strcmp(TEST, "GetContiguousIIndexStartIndices"))      rc += GetContiguousIIndexStartIndices();

    return rc ;
}

int main()
{
    return sphoton_test::main();
}

