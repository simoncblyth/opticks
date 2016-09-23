#pragma once 

#include "OKCORE_API_EXPORT.hh"
#include "OKCORE_HEAD.hh"

class OKCORE_API OpticksBufferSpec 
{
    public:
        static const char* Get(const char* name, bool compute );
    public:
        static const char*  genstep_compute_ ; 
        static const char*  nopstep_compute_ ; 
        static const char*  photon_compute_ ; 
        static const char*  record_compute_ ; 
        static const char*  phosel_compute_ ; 
        static const char*  recsel_compute_ ; 
        static const char*  sequence_compute_ ; 
        static const char*  seed_compute_ ; 
        static const char*  hit_compute_ ; 

        static const char*  genstep_interop_ ; 
        static const char*  nopstep_interop_ ; 
        static const char*  photon_interop_ ; 
        static const char*  record_interop_ ; 
        static const char*  phosel_interop_ ; 
        static const char*  recsel_interop_ ; 
        static const char*  sequence_interop_ ; 
        static const char*  seed_interop_ ; 
        static const char*  hit_interop_ ; 
};

#include "OKCORE_TAIL.hh"


