#pragma once
/**
QCurandStateMonolithic.hh : allocate + create + download + save
=================================================================

* creates states using curand_init with CUDA launchs configured by SLaunchSequence.h
* loading/saving from/to file is handled separately by QRng

The curandState originate on the device as a result of 
calling curand_init and they need to be downloaded and stored
into files named informatively with seeds, counts, offsets etc..

A difficulty is that calling curand_init is a very heavy kernel, 
so currently the below large files are created via multiple launches all 
writing into the single files shown below.  
The old cuRANDWrapper and new QCurandStateMonolithic have exactly the same contents. 

+-----------+---------------+----------------+--------------------------------------------------------------------+
|  num      | bytes (ls)    | filesize(du -h)|  path                                                              |
+===========+===============+================+====================================================================+
|   200M    |  8800000000   |   8.2G         | /home/blyth/.opticks/rngcache/RNG/cuRANDWrapper_200000000_0_0.bin  |
+-----------+---------------+----------------+--------------------------------------------------------------------+
|   100M    |  4400000000   |   4.1G         | /home/blyth/.opticks/rngcache/RNG/cuRANDWrapper_100000000_0_0.bin  |
+-----------+---------------+----------------+--------------------------------------------------------------------+
|    10M    |   440000000   |   420M         | /home/blyth/.opticks/rngcache/RNG/cuRANDWrapper_10000000_0_0.bin   | 
+-----------+---------------+----------------+--------------------------------------------------------------------+
|     3M    |   132000000   |   126M         | /home/blyth/.opticks/rngcache/RNG/cuRANDWrapper_3000000_0_0.bin    |
+-----------+---------------+----------------+--------------------------------------------------------------------+
|     2M    |    88000000   |    84M         | /home/blyth/.opticks/rngcache/RNG/cuRANDWrapper_2000000_0_0.bin    |
+-----------+---------------+----------------+--------------------------------------------------------------------+
|     1M    |    44000000   |    42M         | /home/blyth/.opticks/rngcache/RNG/cuRANDWrapper_1000000_0_0.bin    |
+-----------+---------------+----------------+--------------------------------------------------------------------+

+-----------+---------------+----------------+-------------------------------------------------------------------------------+
|  num      | bytes (ls)    | filesize(du -h)|  path                                                                         |
+===========+===============+================+===============================================================================+
|    10M    |   440000000   |   420M         | /home/blyth/.opticks/rngcache/RNG/QCurandStateMonolithiic_10000000_0_0.bin    |   
+-----------+---------------+----------------+-------------------------------------------------------------------------------+
|     3M    |   132000000   |   126M         | /home/blyth/.opticks/rngcache/RNG/QCurandStateMonolithic_3000000_0_0.bin      |
+-----------+---------------+----------------+-------------------------------------------------------------------------------+
|     1M    |    44000000   |    42M         | /home/blyth/.opticks/rngcache/RNG/QCurandStateMonolithic_1000000_0_0.bin      | 
+-----------+---------------+----------------+-------------------------------------------------------------------------------+


With GPU VRAM of 48G the limit coming from combination of photons and curandStates is about 400M

* curandState item size in the files is 44 bytes which get padded to 48 bytes in curandState type
* dealing with 16.4GB files for 400M states is uncomfortable, so will need to rearrange into multiple files
* chunking into files of 10M states each would correspond to 40 files of 10M states each (420M bytes) 
* with 40-100 files of 10M states each could push to one billion photon launch if had GPU with 100G VRAM 
* also could arrange for just the needed states (in 10M chunks) to be loaded+uploaded 
  depending on configured max photon, which depends on available VRAM 


Decide on max size of photon launches by scaling from 48G for 400M, eg with 8G VRAM::

    In [2]: 8.*400./48.
    Out[2]: 66.66666666666667    ## so you might aim for 60M photons max with 8G VRAM


HMM: 61M proves to be over optimistic for small VRAM, see ~/opticks/notes/max_photon_launch_size_with_8GB_VRAM.rst




WIP:chunked creation, chunk naming, chunked save/load 

See SCurandState.h SCurandChunk.h 

**/

#include <string>
#include <cstdint>
#include "QUDARAP_API_EXPORT.hh"
#include "plog/Severity.h"
#include "SCurandStateMonolithic.hh"

struct qcurandstate ; 
struct SLaunchSequence ; 

struct QUDARAP_API QCurandStateMonolithic
{
    static const plog::Severity LEVEL ; 
    static constexpr const char* EKEY = "QCurandStateMonolithic_SPEC" ; 
    static QCurandStateMonolithic* Create(); 
    static QCurandStateMonolithic* Create(const char* spec); 

    const SCurandStateMonolithic scs ; 
    qcurandstate* h_cs ; 
    qcurandstate* cs ; 
    qcurandstate* d_cs ; 
    SLaunchSequence* lseq ; 

    QCurandStateMonolithic(const SCurandStateMonolithic& scs); 
    void init(); 
    void alloc(); 
    void create(); 
    void download(); 
    void save() const ; 

    std::string desc() const ; 
};
