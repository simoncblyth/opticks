// TEST=OpticksProfileTest om-t

#include "OpticksProfile.hh"
#include "OPTICKS_LOG.hh"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc,argv);
    
    LOG(info) << argv[0] ;


    OpticksProfile* op = new OpticksProfile("OpticksProfileTest") ;


    op->setDir("$TMP/OpticksProfileTest") ; // canonically done from Opticks::configure 

    int tagoffset = 0 ;      

    unsigned long long MB = 1 << 10 << 10 ; 

    new char[MB] ; 
    op->stamp( "red:yellow:purple", tagoffset);
    new char[MB] ; 
    op->stamp( "green:pink:violet", tagoffset);
    new char[MB] ; 
    op->stamp( "blue:cyan:indigo", tagoffset);



    op->save(); 


    return 0 ;
}

/*

Using labels with dots results in "ptree too deep"


2019-05-10 16:09:37.290 FATAL [721] [OpticksProfile::stamp@90] OpticksProfile::stamp red.yellow.purple_0 (0,29377.3,0,121.06)
2019-05-10 16:09:37.290 FATAL [721] [OpticksProfile::stamp@90] OpticksProfile::stamp green.pink.violet_0 (0,0,1.028,1.028)
2019-05-10 16:09:37.290 FATAL [721] [OpticksProfile::stamp@90] OpticksProfile::stamp blue.cyan.indigo_0 (0,0,2.056,1.028)
2019-05-10 16:09:37.290 ERROR [721] [OpticksProfile::save@116]  dir $TMP/OpticksProfileTest name OpticksProfileTest.npy num_stamp 3
terminate called after throwing an instance of 'boost::exception_detail::clone_impl<boost::exception_detail::error_info_injector<boost::property_tree::ini_parser::ini_parser_error> >'
  what():  /tmp/blyth/opticks/OpticksProfileTest/Time.ini: ptree is too deep
Aborted (core dumped)
[blyth@localhost optickscore]$ 


*/


