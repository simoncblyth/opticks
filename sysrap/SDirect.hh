
/**
SDirect
========

Stream redirection experiments.


**/


// http://stackoverflow.com/questions/5419356/redirect-stdout-stderr-to-a-string

#include <iostream>
#include <streambuf>

struct cout_redirect {
    cout_redirect( std::streambuf * new_buffer ) 
        : old( std::cout.rdbuf( new_buffer ) ) 
    { } 

    ~cout_redirect( ) { 
        std::cout.rdbuf( old );
    }   

private:
    std::streambuf * old;
};


struct cerr_redirect {
    cerr_redirect( std::streambuf * new_buffer ) 
        : old( std::cerr.rdbuf( new_buffer ) ) 
    { } 

    ~cerr_redirect( ) { 
        std::cerr.rdbuf( old );
    }   

private:
    std::streambuf * old;
};



struct stream_redirect
{
    // http://wordaligned.org/articles/cpp-streambufs
    stream_redirect(std::ostream & dst, std::ostream & src)
        :   
        src(src), 
        sbuf(src.rdbuf(dst.rdbuf())) 
    {   
    }   

    ~stream_redirect() 
    { 
       src.rdbuf(sbuf); 
    }

private:
    std::ostream & src;
    std::streambuf * const sbuf;
};




