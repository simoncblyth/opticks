#include "Opticks.hh"

#include "BRAP_LOG.hh"
#include "OKCORE_LOG.hh"
#include "PLOG.hh"

int main(int argc, char** argv, char** /*envp*/)
{
    PLOG_(argc, argv);

    BRAP_LOG__ ;
    OKCORE_LOG__ ;

    Opticks ok(argc, argv) ;
    ok.configure();
    ok.Summary();

    if(argc > 1 && strlen(argv[1]) > 0)
    {
        ok.prepareInstallCache(argv[1]);
    }
    else
    {
        ok.prepareInstallCache();
    }

    return 0 ; 
}

/*

Test run with argument /tmp/ores writes 4 identical flag files::

    simon:ores blyth$ ll
    total 32
    -rw-r--r--   1 blyth  wheel  222 Jul 12 21:28 GFlagsSource.ini
    -rw-r--r--   1 blyth  wheel  222 Jul 12 21:28 GFlagsLocal.ini
    -rw-r--r--   1 blyth  wheel  222 Jul 12 21:28 GFlagIndexSource.ini
    -rw-r--r--   1 blyth  wheel  222 Jul 12 21:28 GFlagIndexLocal.ini

    simon:ores blyth$ md5 *.ini
    MD5 (GFlagIndexLocal.ini)  = 2b2b87a16a5232f4f4fa5d4c30599e2b
    MD5 (GFlagIndexSource.ini) = 2b2b87a16a5232f4f4fa5d4c30599e2b
    MD5 (GFlagsLocal.ini)      = 2b2b87a16a5232f4f4fa5d4c30599e2b
    MD5 (GFlagsSource.ini)     = 2b2b87a16a5232f4f4fa5d4c30599e2b

    simon:ores blyth$ head -5 GFlagIndexLocal.ini 
    BOUNDARY_REFLECT=11
    BOUNDARY_TRANSMIT=12
    BULK_ABSORB=4
    BULK_REEMIT=5
    BULK_SCATTER=6

Canonical run without argument writes into resource dir::

    simon:optickscore blyth$ ll /usr/local/opticks/opticksdata/resource/
    total 32
    drwxr-xr-x  3 blyth  staff  102 Jul  9 18:40 OpticksColors
    drwxr-xr-x  4 blyth  staff  136 Jul  9 18:40 GFlags
    drwxr-xr-x  8 blyth  staff  272 Jul 12 13:02 ..
    -rw-r--r--  1 blyth  staff  222 Jul 13 10:53 GFlagsSource.ini
    -rw-r--r--  1 blyth  staff  222 Jul 13 10:53 GFlagsLocal.ini
    -rw-r--r--  1 blyth  staff  222 Jul 13 10:53 GFlagIndexSource.ini
    -rw-r--r--  1 blyth  staff  222 Jul 13 10:53 GFlagIndexLocal.ini
    drwxr-xr-x  8 blyth  staff  272 Jul 13 10:53 .
    simon:optickscore blyth$ 


That is not really the right place either, as entails manual management via 
opticksdata.

A better place would be /usr/local/opticks/cache/ beside the rng 


*/
