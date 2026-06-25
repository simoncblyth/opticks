/**
sreport.cc : Summarize + Present SEvt/NPFold metadata time stamps
=============================================================================

::

    ~/opticks/sysrap/tests/sreport.sh
    ~/opticks/sysrap/tests/sreport.sh grab
    ~/opticks/sysrap/tests/sreport.sh ana


Summarizes SEvt/NPFold metadata time stamps into substamp arrays
grouped by NPFold path prefix. The summary NPFold is presented textually
and saved to allow plotting from python.

+-----+---------------------------------+-------------------------+
| key | SEvt/NPFold path prefix         |  SEvt type              |
+=====+=================================+=========================+
| a   | "//A" eg: //A000 //A001         | Opticks/QSim SEvt       |
+-----+---------------------------------+-------------------------+
| b   | "//B" eg: //B000 //B001         | Geant4/U4Recorder SEvt  |
+-----+---------------------------------+-------------------------+

The tables are presented with row and column labels and the
summary NPFold is saved to DIR_sreport sibling to invoking DIR
which needs to contain SEvt/NPFold folders corresponding to the path prefix.
The use of NPFold::LoadNoData means that only SEvt NPFold/NP
metadata is loaded. Excluding the array data makes the load
very fast and able to handle large numbers of persisted SEvt NPFold.

Usage from source "run" directory creates the report saving into eg ../ALL3_sreport::

    epsilon:~ blyth$ cd /data/blyth/opticks/GEOM/J23_1_0_rc3_ok0/G4CXTest/ALL3
    epsilon:ALL3 blyth$ sreport
    epsilon:ALL3 blyth$ ls -alst ../ALL3_sreport

Usage from report directory loads and presents the report::

    epsilon:ALL3 blyth$ cd ../ALL3_sreport/
    epsilon:ALL3_sreport blyth$ sreport

Note that this means that can rsync just the small report directory
and still be able to present the report and make plots on laptop concerning
run folders with many large arrays left on the server.


Test archiving sreport using indexed folders
----------------------------------------------

::

     export SREPORT_ARCHIVE=/tmp/archive
     mkdir -p $SREPORT_ARCHIVE
     cxs_min.sh report
     cxs_min.sh report  ## repeating duplicates the report


Debugging Notes
-----------------

Debugging this is whacky as its mostly stringstream preparation
so cout/cerr reporting sometimes seems out of place compared to
the report output. For this reason its important to label most
output with where it comes from to speedup understanding+debug.

**/


#include "sreport.h"
#include "sreport_Creator.h"


int main(int argc, char** argv)
{
    const char* CONFIG = U::GetEnv(sreport::sreport__CONFIG,"");
    bool _main = sreport::IsConfig(CONFIG, "main") ;

    char* argv0 = argv[0] ;
    const char* dirp = argc > 1 ? argv[1] : U::PWD() ;
    if(dirp == nullptr) return 0 ;
    bool is_executable_sibling_path = U::IsExecutableSiblingPath( argv0 , dirp ) ;
    bool is_indexed_path = sreport::IsIndexedDirname(dirp);
    bool just_load_report = is_executable_sibling_path || is_indexed_path ;
    // When run from "_sreport" directory OR from an indexed dirname eg "sreport_00000" simply load the report

    std::cout
       << "[sreport.main"
       << " CONFIG [" << ( CONFIG ? CONFIG : "-" ) << "]"
       << " argv0 " << ( argv0 ? argv0 : "-" )
       << " dirp " << ( dirp ? dirp : "-" )
       << " is_executable_sibling_path " << ( is_executable_sibling_path ? "YES" : "NO " )
       << " is_indexed_path " << ( is_indexed_path ? "YES" : "NO " )
       << std::endl
       ;

    if( just_load_report == false )
    {
        U::SetEnvDefaultExecutableSiblingPath("SREPORT_FOLD", argv0, dirp );
        if(_main) std::cout << "[sreport.main : CREATING REPORT " << std::endl ;

        if(_main) std::cout << "[sreport.main : creator " << std::endl ;
        sreport_Creator creator(dirp, CONFIG);
        if(_main) std::cout << "]sreport.main : creator " << std::endl ;
        if(_main) std::cout << "[sreport.main : creator.desc " << std::endl ;
        if(_main) std::cout << creator.desc() ;
        if(_main) std::cout << "]sreport.main : creator.desc " << std::endl ;
        if(!creator.fold_valid) return 1 ;

        sreport* report = creator.report ;
        if(_main) std::cout << "[sreport.main : report.desc " << std::endl ;
        std::cout << report->desc() ;
        if(_main) std::cout << "]sreport.main : report.desc " << std::endl ;
        report->save("$SREPORT_FOLD");
        if(_main) std::cout << "]sreport.main : CREATED REPORT " << std::endl ;

        const char* SREPORT_ARCHIVE = U::GetEnv("SREPORT_ARCHIVE", nullptr);
        if(SREPORT_ARCHIVE)
        {
            std::cout << "[sreport.main : save_into_archive [" << SREPORT_ARCHIVE << "]\n" ;
            report->save_into_archive(SREPORT_ARCHIVE);
            std::cout << "]sreport.main : save_into_archive [" << SREPORT_ARCHIVE << "]\n" ;
        }


        if(getenv("CHECK") != nullptr )
        {
            std::cout << "[sreport.main : CHECK LOADED REPORT " << std::endl ;
            sreport* report2 = sreport::Load("$SREPORT_FOLD") ;
            std::cout << report2->desc() ;
            std::cout << "]sreport.main : CHECK LOADED REPORT " << std::endl ;
        }

    }
    else
    {
        std::cout << "[sreport.main : LOADING REPORT " << std::endl ;
        sreport* report = sreport::Load(dirp) ;
        std::cout << report->desc() ;
        std::cout << "]sreport.main : LOADED REPORT " << std::endl ;
    }

    std::cout
       << "]sreport.main"
       << std::endl
       ;

    return 0 ;
}

