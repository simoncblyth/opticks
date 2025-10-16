#pragma once
/**
spho.h : photon labelling used by genstep collection
========================================================

After cfg4/CPho

isSameLineage
    does not require the same reemission generation

isIdentical
    requires isSameLineage and same reemission generation

NB spho lacks gentype, to get that must reference corresponding sgs struct using the gs index

NB having reemission generations larger than zero DOES NOT mean the
photon originally came from scintillaton.
For example in a case where no photons are coming from scint,
reemission of initially Cerenkov photons may still happen,
resulting in potentially multiple reemission generations.

**/

#include <array>
#include <string>

struct spho_uchar4 { unsigned char x,y,z,w ; };

union spho_uuc4 {
    unsigned    u   ;
    spho_uchar4 uc4 ;
};




struct spho // WIP: unsigned ?
{
    static constexpr const int N = 4 ;

    int gs ; // 0-based genstep index within the event
    int ix ; // 0-based photon index within the genstep
    int id ; // 0-based photon identity index within the event

    spho_uchar4 uc4 ;
    // uc4.x : gen : 0-based reemission index incremented at each reemission
    // uc4.y : eph : eg junoSD_PMT_v2::ProcessHits eph enumeration
    // uc4.z : ext : unused
    // uc4.w : photon point flag TO/BT/BR/SC/AB/SD/SR/... etc

    unsigned uc4packed() const ;

    int gen() const ;
    int eph() const ;
    int ext() const ;
    int flg() const ;

    void set_gen(int gn) ;
    void set_eph(int ep) ;
    void set_ext(int ex) ;
    void set_flg(int fg) ;

    static spho MakePho(int gs_, int ix_, int id_ );
    static spho Fabricate(int track_id);
    static void FabricateInplace( int track_id, int* label );
    void set_fabricated_(int track_id );

    static spho Placeholder() ;

    bool isSameLineage(const spho& other) const { return gs == other.gs && ix == other.ix && id == other.id ; }
    bool isIdentical(const spho& other) const { return isSameLineage(other) && uc4.x == other.uc4.x ; }

    bool isPlaceholder() const { return gs == -1 ; }
    bool isDefined() const {     return gs != -1 ; }

    spho make_nextgen() const ; // formerly make_reemit
    std::string desc() const ;

    const int* cdata() const ;
    int* data();
    void serialize( std::array<int, 4>& a ) const ;
    void load( const std::array<int, 4>& a );

};


#include <cassert>
#include <sstream>
#include <iomanip>

inline unsigned spho::uc4packed() const
{
    spho_uuc4 uuc4 ;
    uuc4.uc4 = uc4 ;
    return uuc4.u ;
}

inline int spho::gen() const { return int(uc4.x); }
inline int spho::eph() const { return int(uc4.y); }
inline int spho::ext() const { return int(uc4.z); }
inline int spho::flg() const { return int(uc4.w); }

inline void spho::set_gen(int gn) { uc4.x = (unsigned char)(gn) ; }
inline void spho::set_eph(int ep) { uc4.y = (unsigned char)(ep) ; }
inline void spho::set_ext(int ex) { uc4.z = (unsigned char)(ex) ; }
inline void spho::set_flg(int fg) { uc4.w = (unsigned char)(fg) ; }


inline spho spho::MakePho(int gs_, int ix_, int id_) // static
{
    spho ph = {gs_, ix_, id_, {0,0,0,0} } ;
    return ph ;
}
/**
spho::Fabricate
---------------

*Fabricate* is not normally used, as C+S photons are always
labelled at generation by U4::GenPhotonEnd

However as a workaround for torch/input photons that lack labels
this method is used from U4Recorder::PreUserTrackingAction_Optical
to provide a standin label based only on a 0-based track_id.

**/
inline spho spho::Fabricate(int track_id) // static
{
    assert( track_id >= 0 );
    spho fab = {0, track_id, track_id, {0,0,0,0} };
    return fab ;
}

inline void spho::FabricateInplace( int track_id, int* label )
{
    spho& q = reinterpret_cast<spho&>(*label);
    q.set_fabricated_(track_id);
}

inline void spho::set_fabricated_(int track_id )
{
    assert( track_id >= 0 );
    gs = 0 ;
    ix = track_id ;
    id = track_id ;
    set_gen(0);
    set_eph(0);
    set_ext(0);
    set_flg(0);
}


inline spho spho::Placeholder() // static
{
    spho inv = {-1, -1, -1, {0,0,0,0} };
    return inv ;
}
inline spho spho::make_nextgen() const
{
    spho nextgen = *this ;
    nextgen.uc4.x += 1 ;
    return nextgen ;
}

inline std::string spho::desc() const
{
    std::stringstream ss ;
    ss << "spho" ;
    if(isPlaceholder())
    {
        ss << " isPlaceholder " ;
    }
    else
    {
        ss << " (gs:ix:id:gn "
           << std::setw(3) << gs
           << std::setw(4) << ix
           << std::setw(5) << id
           << "["
           << std::setw(3) << int(uc4.x) << ","
           << std::setw(3) << int(uc4.y) << ","
           << std::setw(3) << int(uc4.z) << ","
           << std::setw(3) << int(uc4.w)
           << "]"
           << ")"
           ;
    }
    std::string s = ss.str();
    return s ;
}



inline int* spho::data()
{
    return &gs ;
}
inline const int* spho::cdata() const
{
    return &gs ;
}
inline void spho::serialize( std::array<int, 4>& a ) const
{
    assert( a.size() == N );
    const int* ptr = cdata() ;
    for(int i=0 ; i < N ; i++ ) a[i] = ptr[i] ;
}
inline void spho::load( const std::array<int, 4>& a )
{
    assert( a.size() == N );
    int* ptr = data() ;
    for(int i=0 ; i < N ; i++ ) ptr[i] = a[i] ;
}




inline std::ostream& operator<<(std::ostream& os, const spho& p)
{
    os << p.desc() ;
    return os;
}



