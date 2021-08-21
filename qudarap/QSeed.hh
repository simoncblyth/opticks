#pragma once

struct quad6 ;
template <typename T> struct QBuf ; 
#include "QUDARAP_API_EXPORT.hh"

/**
QSeed
======

The photon seed buffer is a device buffer containing integer indices referencing 
into the genstep buffer. The seeds provide the association between the photon 
and the genstep required to generate it.

**/

struct QUDARAP_API QSeed
{
    static QBuf<int>* CreatePhotonSeeds(QBuf<quad6>* gs); 

    // testing 
    static void ExpectedSeeds(std::vector<int>& seeds,  const std::vector<int>& counts );
    static QBuf<quad6>* UploadFakeGensteps(const std::vector<int>& counts) ; 
    static int    CompareSeeds( const std::vector<int>& seeds, const std::vector<int>& xseeds ); 

};



