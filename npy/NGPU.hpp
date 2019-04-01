#pragma once

#include <string>
#include <vector>
#include "NPY_API_EXPORT.hh"

template <typename T> class NPY ; 

/**
NGPU
=======

NGPU is used to record GPU buffer uploads, recording 
the number of bytes and name/owner/note context information.
As these strings are saved into an "unsigned long long" 
NPY buffer characters beyond 8 are truncated.

OGLRap and OptiXRap are instrumented to record the buffer 
usage.

Options to configure  usage are:

--gpumonpath
    path of the NPY buffer

--gpumon 
    switch on the saving and dumping, which are done at 
    OpticksHub::cleanup 


Prior saved monitor buffers can be viewed using NGPUTest::

    epsilon:npy blyth$ NGPUTest /tmp/blyth/opticks/GPUMonPath.npy
    2018-07-04 20:03:13.358 INFO  [1248883] [NGPU::dump@114] NGPU::dump num_records 94 num_bytes 131781536
           itransfo       nrm.....       RBuf:upl :              64 :       0.00
           vertices       nrm.....       RBuf:upl :         2453568 :       2.45
           colors..       nrm.....       RBuf:upl :         2453568 :       2.45
           normals.       nrm.....       RBuf:upl :         2453568 :       2.45
           indices.       nrm.....       RBuf:upl :         4844544 :       4.84
           itransfo       nrmvec..       RBuf:upl :              64 :       0.00
           vertices       nrmvec..       RBuf:upl :         2453568 :       2.45
           colors..       nrmvec..       RBuf:upl :         2453568 :       2.45
           ...
           vertexBu       OGeo5-1.       cibGBuf. :           17688 :       0.02
           indexBuf       OGeo5-1.       cibGBuf. :           35136 :       0.04
           OBndLib.       OPropLib       OScene.. :          614016 :       0.61
           vertices       tex.....       RBuf:upl :              48 :       0.00
           colors..       tex.....       RBuf:upl :              48 :       0.00
           normals.       tex.....       RBuf:upl :              48 :       0.00
           texcoord       tex.....       RBuf:upl :              32 :       0.00
           indices.       tex.....       RBuf:upl :              24 :       0.00
           vpos....       axis_att       Rdr:upl. :             144 :       0.00
           vpos....       genstep_       Rdr:upl. :              96 :       0.00
           vpos....       nopstep_       Rdr:upl. :               0 :       0.00
           vpos....       photon_a       Rdr:upl. :         6400000 :       6.40
           rpos....       record_a       Rdr:upl. :        16000000 :      16.00
           phis....       sequence       Rdr:upl. :         1600000 :       1.60
           psel....       phosel_a       Rdr:upl. :          400000 :       0.40
           rsel....       recsel_a       Rdr:upl. :         4000000 :       4.00

                       TOTALS in bytes, Mbytes :  :       131781536 :     131.78
**/

struct NPY_API NGPURecord
{
    typedef unsigned long long ULL ; 
    NGPURecord( 
        ULL name_, 
        ULL owner_, 
        ULL note_, 
        ULL num_bytes_);

    char name[8+1] ;   
    char owner[8+1] ;   
    char note[8+1] ;   
    unsigned long long num_bytes ;   

    void set_name(const char* name_);
    void set_owner(const char* owner_);
    void set_note(const char* note_);

    std::string desc() const  ;
};


struct NPY_API NGPU 
{
    static NGPU* fInstance ; 
    static NGPU* GetInstance() ; 
    static NGPU* Load(const char* path) ; 

    typedef unsigned long long ULL ; 
    NPY<ULL>* recs ; 
    std::vector<NGPURecord> records ; 

    NGPU(); 
    NGPU(NPY<ULL>* recs_);

    void import(); 

    void add( unsigned long long num_bytes, const char* name, const char* owner, const char* note=NULL ); 
    void saveBuffer(const char* path); 
    unsigned long long getNumBytes() const ;

    void dump(const char* msg="NGPU::dump") const ;



};

