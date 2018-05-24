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
        LOG(warning) << "WRITING TO MANUAL PATH IS JUST FOR TESTING" ; 
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


Formerly wrote into IDPATH and then /usr/local/opticks/opticksdata/resource/
but they are not the right places, as not geometry related and no need 
to require manual management via opticksdata.

OKC flags Ended up in a more appropriate place as
siblings to PTX and RNG which have same lifecycle, 
ie born at or just after installation

   /usr/local/opticks/installcache/OKC
   /usr/local/opticks/installcache/PTX
   /usr/local/opticks/installcache/RNG


*/
