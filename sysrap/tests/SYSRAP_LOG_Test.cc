#include <cassert>
#include "PLOG.hh"
#include "SArgs.hh"

#include "SYSRAP_LOG.hh"


void check(const char* msg)
{
    LOG(trace) << msg ; 
    LOG(verbose) << msg ; 
    LOG(info) << msg ; 
    LOG(debug) << msg ; 
    LOG(warning) << msg ; 
    LOG(error) << msg ; 
    LOG(fatal) << msg ; 
}


void test_standard_usage(int argc, char** argv)
{
    PLOG_(argc, argv);
    SYSRAP_LOG__ ; 

    assert(PLOG::instance);

    check("local");
    SYSRAP_LOG::Check("calling SYSRAP lib from test main");
}


int main(int, char** )
{
    std::vector<std::string> tt  ; 
    tt.push_back( "--trace" );
    tt.push_back( "--verbose" );
    tt.push_back( "--debug" );
    tt.push_back( "--info" );
    tt.push_back( "--warning" );
    tt.push_back( "--error" );
    tt.push_back( "--fatal" );

    tt.push_back( "--TRACE" );
    tt.push_back( "--VERBOSE" );
    tt.push_back( "--DEBUG" );
    tt.push_back( "--INFO" );
    tt.push_back( "--WARNING" );
    tt.push_back( "--ERROR" );
    tt.push_back( "--FATAL" );

    tt.push_back( "--sysrap trace" );
    tt.push_back( "--sysrap verbose" );
    tt.push_back( "--sysrap debug" );
    tt.push_back( "--sysrap info" );
    tt.push_back( "--sysrap warning" );
    tt.push_back( "--sysrap error" );
    tt.push_back( "--sysrap fatal" );

    tt.push_back( "--SYSRAP trace" );
    tt.push_back( "--SYSRAP verbose" );
    tt.push_back( "--SYSRAP debug" );
    tt.push_back( "--SYSRAP info" );
    tt.push_back( "--SYSRAP warning" );
    tt.push_back( "--SYSRAP error" );
    tt.push_back( "--SYSRAP fatal" );

    tt.push_back( "--SYSRAP TRACE" );
    tt.push_back( "--SYSRAP VERBOSE" );
    tt.push_back( "--SYSRAP DEBUG" );
    tt.push_back( "--SYSRAP INFO" );
    tt.push_back( "--SYSRAP WARNING" );
    tt.push_back( "--SYSRAP ERROR" );
    tt.push_back( "--SYSRAP FATAL" );


    const char* fallback = "info" ; 
    const char* prefix = "SYSRAP" ; 

    for(unsigned j=0 ; j < 2 ; j++)
    {

    for(unsigned i=0 ; i < tt.size() ; i++)
    {
        std::string t = tt[i]; 

        SArgs* a = new SArgs(t) ;

        const char* uprefix = j == 0 ? NULL : prefix ; 

        PLOG* pl = new PLOG(a->argc, a->argv, fallback, uprefix );

        std::stringstream ss ; 
        ss << "PLOG(..," << fallback ; 
        if(j==1) ss << "," << uprefix ;
        ss << ")" ;

        std::string label = ss.str();

        std::cout << std::setw(50) << t 
                  << std::setw(30) << label
                  << std::setw(5) << pl->level 
                  << std::setw(10) << pl->name() 
                  << std::endl
                  ; 
    } 
    }

    return 0 ; 
}
