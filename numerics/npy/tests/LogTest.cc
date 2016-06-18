
#include "BLog.hh"


/*
void init_logging(int argc, char** argv)
{
    boost::log::core::get()->set_filter
    (
        boost::log::trivial::severity >= BLog::FilterLevel(argc, argv) 
    );
}
*/


int main(int argc, char** argv)
{
    BLOG(argc, argv);   

    //init_logging(argc, argv);

    //BLog bl(argc, argv);
    //bl.setDir("/tmp"); 
    //BLog::setFilter("info");

    LOG(trace) << argv[0] ; 
    LOG(debug) << argv[0] ; 
    LOG(info) << argv[0] ; 
    LOG(warning) << argv[0] ; 
    LOG(error) << argv[0] ; 
    LOG(fatal) << argv[0] ; 

    return 0 ;
}
