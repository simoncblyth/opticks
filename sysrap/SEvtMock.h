#pragma once

#include "squad.h"
#include "NPX.h"

struct SEvtMock
{
    std::vector<quad6>   genstep ;
    const NP*            hc ;
    SEvtMock();

    void load_genstep(const char* path);

    void   beginOfEvent(int eventID);
    NP*    makeGenstepArrayFromVector() const ;   // formerly misnamed getGenstepArray
    void setHit( const NP* hc );


    static SEvtMock* Get_EGPU();
    static SEvtMock* Get(int idx);
    static SEvtMock* Create(int idx);

    enum { MAX_INSTANCE = 2 } ;
    enum { EGPU, ECPU };
    inline static std::array<SEvtMock*, MAX_INSTANCE> INSTANCES = {nullptr, nullptr} ;

};

inline SEvtMock::SEvtMock()
    :
    hc(nullptr)
{
}


inline void SEvtMock::load_genstep(const char* path)
{
    NP* gs = NP::Load(path);
    assert(gs);
    genstep.resize(gs->shape[0]);
    gs->write<float>((float*)genstep.data());
}


inline void SEvtMock::beginOfEvent(int eventID)
{
    std::cout << "SEvtMock::beginOfEvent " << eventID << "\n" ;
}

inline NP* SEvtMock::makeGenstepArrayFromVector() const
{
    return NPX::ArrayFromData<float>( (float*)genstep.data(), int(genstep.size()), 6, 4 ) ;
}

inline void SEvtMock::setHit( const NP* hc_ )
{
    hc = hc_ ;
}


inline SEvtMock* SEvtMock::Get_EGPU()
{
    return Get(EGPU);
}
inline SEvtMock* SEvtMock::Get(int idx)
{
    assert( idx == 0 || idx == 1 );
    return INSTANCES[idx] ;
}


inline SEvtMock* SEvtMock::Create(int idx)  // static
{
    assert( idx == 0 || idx == 1);
    SEvtMock* sev = new SEvtMock ;
    INSTANCES[idx] = sev  ;
    assert( Get(idx) == sev );
    return sev  ;
}






