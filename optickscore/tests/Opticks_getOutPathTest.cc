#include "OPTICKS_LOG.hh"
#include "Opticks.hh"

void test_getOutPath(const Opticks* ok)
{
    const char* namestem = "namestem" ; 
    for(int idx=-1 ; idx < 10 ; idx++ )
    {    
        const char* outpath = ok->getOutPath( namestem, ".jpg", idx ); 
        std::cout << outpath << std::endl ; 
    }
}

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc,argv);
    
    LOG(info) << argv[0] ;

    Opticks ok(argc, argv);
    ok.configure();

    test_getOutPath(&ok); 
    ok.setOutDir("$TMP/CSGOptiX"); 
    test_getOutPath(&ok) ;

    return 0 ; 
}


/*
Commandline option trumps evar::

    epsilon:optickscore blyth$ OUTDIR=evarout Opticks_getOutPathTest --nameprefix nameprefix_ --outdir optionoutdir
    2021-05-12 11:10:11.763 INFO  [10396867] [main@8] Opticks_getOutPathTest
    optionoutdir/nameprefix_namestem.jpg
    optionoutdir/nameprefix_namestem00000.jpg
    optionoutdir/nameprefix_namestem00001.jpg
    optionoutdir/nameprefix_namestem00002.jpg
    optionoutdir/nameprefix_namestem00003.jpg

*/
