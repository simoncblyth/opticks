// Copyright Vladimir Prus 2002-2004.
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

/* The simplest usage of the library.
 */

#include <boost/program_options.hpp>
namespace po = boost::program_options;

#include <iostream>
#include <iterator>
using namespace std;

int main(int ac, char* av[])
{

    string ipath;
    string opath;
    int compression(0) ; 

    try {

        po::options_description desc("Allowed options");
        desc.add_options()
            ("help", "produce help message")
            ("compression", po::value(&compression), "set compression level")
            ("input,i",  po::value(&ipath), "pathname for input")
            ("output,o", po::value(&opath), "pathname for output")
        ;

	po::positional_options_description p;
	p.add("input", -1);

        po::variables_map vm;        
        po::store(po::command_line_parser(ac, av).options(desc).positional(p).run(), vm);
        po::notify(vm);    

        if (vm.count("help")) {
            cout << desc << "\n";
            return 1;
        }


	cout << "compression " << compression << " opath " << opath << " ipath " << ipath << endl ;

        cout << "ac" << ac << endl ;

	/*
        if (vm.count("compression")) {
            cout << "Compression level was set to " 
                 << vm["compression"].as<int>() << ".\n";
        } else {
            cout << "Compression level was not set.\n";
        }
	*/
    }
    catch(exception& e) {
        cerr << "error: " << e.what() << "\n";
        return 1;
    }
    catch(...) {
        cerr << "Exception of unknown type!\n";
    }

    return 0;
}
