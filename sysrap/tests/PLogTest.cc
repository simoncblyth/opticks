#include <plog/Log.h>
#include <plog/Appenders/ColorConsoleAppender.h>

// translate from boost log levels to plog 
#define fatal plog::fatal
#define error plog::error
#define warning plog::warning
#define info plog::info
#define debug plog::debug
#define trace plog::verbose

using namespace plog ; 

int main(int, char** argv)
{

    static plog::ColorConsoleAppender<plog::TxtFormatter> consoleAppender;
    plog::init(plog::verbose, &consoleAppender);

    //plog::init(plog::debug, "PLogTest.txt");

/*
    LOG(plog::fatal) << argv[0]  ;
    LOG(plog::error) << argv[0]  ;
    LOG(plog::warning) << argv[0]  ;
    LOG(plog::info) << argv[0]  ;
    LOG(plog::debug) << argv[0]  ;
    LOG(plog::verbose) << argv[0]  ;
*/

    LOG(fatal) << argv[0]  ;
    LOG(error) << argv[0]  ;
    LOG(warning) << argv[0]  ;
    LOG(info) << argv[0]  ;
    LOG(debug) << argv[0]  ;
    LOG(trace) << argv[0]  ;


    if(1) LOG(info) << argv[0] << " if-LOG can can cause dangling else problem with some versions of plog " ;




    return 0 ; 
}
