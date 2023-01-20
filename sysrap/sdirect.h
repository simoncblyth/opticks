#pragma once
/**
sdirect.h : Stream redirection to silence noisy code unless VERBOSE is defined
================================================================================

Refs:

* http://stackoverflow.com/questions/5419356/redirect-stdout-stderr-to-a-string
* http://wordaligned.org/articles/cpp-streambufs


Usage::

    std::stringstream coutbuf;
    std::stringstream cerrbuf;
    {
        sdirect::cout_(coutbuf.rdbuf());
        sdirect::cerr_(cerrbuf.rdbuf());
   
        m_dc = new DetectorConstruction ;  // noisy code
   }
   std::string out = coutbuf.str();
   std::string err = cerrbuf.str();
   bool verbose = getenv("VERBOSE") != nullptr ;  
   std::cout << sdirect::OutputMessage("DetectorConstruction", out, err, verbose );


**/

#include <iostream>
#include <iomanip>
#include <streambuf>
#include <sstream>
#include <string>

namespace sdirect
{
    struct cout_ 
    {
        std::streambuf* old;

        cout_( std::streambuf* newbuf ) 
           : 
           old( std::cout.rdbuf(newbuf) ) 
        {} 

        ~cout_() 
        { 
            std::cout.rdbuf( old );
        }   

    };

    struct cerr_ 
    {
        std::streambuf* old;

        cerr_( std::streambuf* newbuf ) 
           : 
           old( std::cerr.rdbuf(newbuf) ) 
        {} 

        ~cerr_() 
        { 
            std::cerr.rdbuf( old );
        }   
    };

    // allows code expecting to write to file to 
    // be tricked into writing to a string 
    struct ostream_
    {
        std::ostream& src;
        std::streambuf * const sbuf;

        ostream_(std::ostream& dst, std::ostream& src)
            :   
            src(src), 
            sbuf(src.rdbuf(dst.rdbuf())) 
        {   
        }   

        ~ostream_() 
        { 
            src.rdbuf(sbuf); 
        }
    };


    inline std::string OutputMessage(const char* msg, const std::string& out, const std::string& err, bool verbose )  // static
    {
        std::stringstream ss ;

        ss << std::left << std::setw(30) << msg << std::right
           << " yielded chars : "
           << " cout " << std::setw(6) << out.size()
           << " cerr " << std::setw(6) << err.size()
           << " : set VERBOSE to see them "
           << std::endl
           ;   

        if(verbose)
        {   
            ss << "cout[" << std::endl << out << "]" << std::endl  ;
            ss << "cerr[" << std::endl << err << "]" << std::endl  ;
        }   
        std::string s = ss.str();
        return s ; 
    }
}


