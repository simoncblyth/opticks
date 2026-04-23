#pragma once

#include "squad.h"
#include "NPX.h"

struct SEvtMock
{
    std::vector<quad6>   genstep ;

    void load_genstep(const char* path);


    void   beginOfEvent(int eventID);
    NP*    makeGenstepArrayFromVector() const ;   // formerly misnamed getGenstepArray
};


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



