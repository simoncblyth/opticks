#pragma once

#include "NPX.h"
#include "NPFold.h"
#include "spath.h"
#include "sdomain.h"

struct swater_RINDEX
{
    static constexpr const char* n_path = "$OPTICKS_PREFIX/include/SysRap/RINDEX_Water_Hale_N.txt" ;
    static constexpr const char* k_path = "$OPTICKS_PREFIX/include/SysRap/RINDEX_Water_Hale_K.txt" ;
    static NP* N();
    static NP* K();
    static NP* Load(const char* path);

    static NPFold* Serialize();
    static int Save(const char* fold="$FOLD");

};

inline NP* swater_RINDEX::N(){ return Load(n_path);}
inline NP* swater_RINDEX::K(){ return Load(k_path);}

inline NP* swater_RINDEX::Load(const char* _path)
{
    const char* path = spath::Resolve(_path);
    NP* prop = NPX::PLoad<double>(path);
    prop->pscale<double>(1000., 0); // um [micrometers] to nm

    NP* en_prop = sdomain::ConvertWavelengthToEnergy(prop,true);
    return en_prop ;
}

inline NPFold* swater_RINDEX::Serialize()
{
    NPFold* f = new NPFold ;
    f->add("N", N());
    f->add("K", K());
    return f ;
}

inline int swater_RINDEX::Save(const char* fold)
{
    NPFold* f = Serialize();
    f->save(fold);
    return 0 ;
}






