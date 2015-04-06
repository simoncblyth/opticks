// http://stackoverflow.com/questions/12755755/want-to-allow-options-to-be-specified-multiple-times-when-using-boost-program-op
// clang++ -I/opt/local/include -L/opt/local/lib -lboost_program_options-mt composing.cpp -o /tmp/composing

/*

delta:bpo blyth$ /tmp/composing --debug 1 --debug 2 --debug 3
got 3 debug values
using last value of 3
delta:bpo blyth$ 


*/



#include <boost/assign/list_of.hpp>
#include <boost/program_options.hpp>
#include <boost/version.hpp>

#include <iostream>

int
main( int argc, char** argv )
{
    namespace po = boost::program_options;

    po::options_description desc("Options");

    typedef std::vector<unsigned> DebugValues;
    DebugValues debug;
    desc.add_options()
        ("help,h", "produce help message")
        ("debug", po::value<DebugValues>(&debug)->default_value(boost::assign::list_of(0), "0")->composing(), "set debug level")

        ;

    po::variables_map vm;
    try {
        const po::positional_options_description p; // note empty positional options
        po::store(
                po::command_line_parser( argc, argv).
                          options( desc ).
//                          positional( p ).
                          run(),
                          vm
                          );
        po::notify( vm );

        if ( vm.count("help") ) {
            std::cout << desc << "\n";
            std::cout << "boost version: " << BOOST_LIB_VERSION << std::endl;
            return 0;
        }
    } catch ( const boost::program_options::error& e ) {
        std::cerr << e.what() << std::endl;
    }

    std::cout << "got " << debug.size() << " debug values" << std::endl;
    if ( !debug.empty() ) {
        DebugValues::const_iterator value( debug.end() );
        std::advance( value, -1 );
        std::cout << "using last value of " << *value << std::endl;
    }
}
