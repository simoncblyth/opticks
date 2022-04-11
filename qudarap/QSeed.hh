#pragma once

//struct quad6 ;
template <typename T> struct QBuf ; 
struct QEvent ; 

#include "QUDARAP_API_EXPORT.hh"

/**
QSeed
======

The photon seed buffer is a device buffer containing integer indices referencing 
into the genstep buffer. The seeds provide the association between the photon 
and the genstep required to generate it.

TODO: All event releated uploading/downloading and pointers 
should be controlled from one place : QEvent (not here)

**/

struct QUDARAP_API QSeed
{
    // on GPU seeding using thrust 
    static QBuf<int>* CreatePhotonSeeds(QBuf<float>* gs); 

    static void CreatePhotonSeeds( QEvent* evt ); 

    // testing 
    static void ExpectedSeeds(std::vector<int>& seeds,  unsigned& total, const std::vector<int>& counts );
    static int  CompareSeeds( const std::vector<int>& seeds, const std::vector<int>& xseeds ); 
};



