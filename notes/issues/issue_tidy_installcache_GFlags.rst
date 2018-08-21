issue_tidy_installcache_GFlags
=================================


Why are there four versions of the same thing ?::

    epsilon:ana blyth$ ll /usr/local/opticks/installcache/OKC/
    total 32
    drwxr-xr-x  5 blyth  staff  160 Apr  5 11:02 ..
    -rw-r--r--  1 blyth  staff  274 Apr  5 11:02 GFlagsSource.ini
    -rw-r--r--  1 blyth  staff  274 Apr  5 11:02 GFlagsLocal.ini
    -rw-r--r--  1 blyth  staff  274 Apr  5 11:02 GFlagIndexSource.ini
    drwxr-xr-x  6 blyth  staff  192 Apr  5 11:02 .
    -rw-r--r--  1 blyth  staff  274 Apr  5 11:02 GFlagIndexLocal.ini

    epsilon:OKC blyth$ md5 *
    MD5 (GFlagIndexLocal.ini) = 6bf1611da4da4c32928b1b2b3038374d
    MD5 (GFlagIndexSource.ini) = 6bf1611da4da4c32928b1b2b3038374d
    MD5 (GFlagsLocal.ini) = 6bf1611da4da4c32928b1b2b3038374d
    MD5 (GFlagsSource.ini) = 6bf1611da4da4c32928b1b2b3038374d
    epsilon:OKC blyth$ 


Also not updated for PRIMARY_SOURCE.::

    epsilon:ana blyth$ cat /usr/local/opticks/installcache/OKC/GFlagsSource.ini 
    BOUNDARY_REFLECT=11
    BOUNDARY_TRANSMIT=12
    BULK_ABSORB=4
    BULK_REEMIT=5
    BULK_SCATTER=6
    CERENKOV=1
    EMITSOURCE=19
    FABRICATED=16
    G4GUN=15
    MACHINERY=18
    MISS=3
    NAN_ABORT=14
    NATURAL=17
    SCINTILLATION=2
    SURFACE_ABSORB=8
    SURFACE_DETECT=7
    SURFACE_DREFLECT=9
    SURFACE_SREFLECT=10
    TORCH=13


Also, are getting abbreviations from opticksdata which is an anacronism : should be in installcache:: 

    epsilon:thrustrap blyth$ cat /usr/local/opticks/opticksdata/resource/GFlags/abbrev.json 
    {
        "CERENKOV":"CK",
        "SCINTILLATION":"SI",
        "TORCH":"TO",
        "MISS":"MI",
        "BULK_ABSORB":"AB",
        "BULK_REEMIT":"RE", 
        "BULK_SCATTER":"SC",    
        "SURFACE_DETECT":"SD",
        "SURFACE_ABSORB":"SA",      
        "SURFACE_DREFLECT":"DR",
        "SURFACE_SREFLECT":"SR",
        "BOUNDARY_REFLECT":"BR",
        "BOUNDARY_TRANSMIT":"BT",
        "NAN_ABORT":"NA"
    }




