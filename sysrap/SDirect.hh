/*
 * Copyright (c) 2019 Opticks Team. All Rights Reserved.
 *
 * This file is part of Opticks
 * (see https://bitbucket.org/simoncblyth/opticks).
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); 
 * you may not use this file except in compliance with the License.  
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software 
 * distributed under the License is distributed on an "AS IS" BASIS, 
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
 * See the License for the specific language governing permissions and 
 * limitations under the License.
 */


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




