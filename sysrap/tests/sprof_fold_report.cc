/**
sprof_fold_report.cc : Summarize + Present SEvt/NPFold metadata profile stamps 
==============================================================================

Summarizes SEvt/NPFold metadata profile time/VM/RS stamps into substamp arrays 
grouped by NPFold path prefix. The summary NPFold is presented textually 
and saved to allow plotting from python. 

The machinery is similar to sstampfold_report.cc however some differences:

1. have extra VM and RSS memory values in addition to time stamps
2. with profile info are more interested in changes between 
   events (eg memory leaks) and are not concerned with changes 
   within events which are more relevant to sstampfold_report

TODO
----- 

* ELIMINATE THIS BY COMBINING INTO sreport.cc 

* implement C++ textual presentation of the summary info :
  can start the same as sstampfold but in triplicate

* bring over the plotting machinery from ~/np/tests/NPFold_profile_test.py 



**/

#include <cstdlib>
#include "sprof_fold.h"


int main(int argc, char** argv)
{
    const char* dirp = argc > 1 ? argv[1] : U::PWD() ;   
    if(dirp == nullptr) return 0 ; 
    U::SetEnvDefaultExecutableSiblingPath("FOLD", argv[0], dirp );

    bool VERBOSE = getenv("VERBOSE") != nullptr ; 
 
    NPFold* f = NPFold::LoadNoData(dirp); 

    std::cout 
        << "sprof_fold_report"
        << std::endl
        << "NPFold::LoadNoData(\"" << dirp << "\")" 
        << std::endl
        ;

    if(VERBOSE) std::cout 
        << "[sprof_fold_report.VERBOSE "
        << std::endl
        << f->desc()
        << std::endl
        << "]sprof_fold_report.VERBOSE "
        << std::endl
        ; 

    NPFold* smry = f->subfold_summary(sprof_fold::STAMP_KEY, "a://p", "b://n"); 
    std::cout << smry->desc() << std::endl ;

    const NPFold* a = smry->find_subfold("a"); 
    const NPFold* b = smry->find_subfold("b"); 

    sprof_fold apr(a, "apr"); 
    sprof_fold bpr(b, "bpr"); 

    std::cout << apr.desc() ; 
    std::cout << bpr.desc() ; 

    smry->save_verbose("$FOLD"); 

    return 0 ; 
}

