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



#include <iomanip>
#include <iostream>
#include <string>
#include <vector>


void dump( bool b0, bool b1, float f, double d, int i, unsigned u, std::vector<float>& v)
{
    std::cout   
        << std::setw(10) << " b0 " << " : " << b0 << std::endl 
        << std::setw(10) << " b1 " << " : " << b1 << std::endl 
        << std::setw(10) << " f " << " : " << f << std::endl 
        << std::setw(10) << " d " << " : " << d << std::endl 
        << std::setw(10) << " i " << " : " << i << std::endl 
        << std::setw(10) << " u " << " : " << u << std::endl 
        << std::setw(10) << " v " << " : " << v.size() << std::endl 
        ;
}

struct Obj
{
    Obj()
       :
       _b0(false), 
       _b1(true), 
       _f(1.123f), 
       _d(1.123),  
       _i(-5),
       _u(6),
       _c(-128), 
       _uc(255),
       _cc("hello const char*"), 
       _s("hello std::string") 
    {
    }

    bool _b0 ; 
    bool _b1 ; 
    float _f ; 
    double _d ; 
    int _i ;
    unsigned _u ;
    char _c ; 
    unsigned char _uc ; 
    const char* _cc  ; 
    std::string _s    ; 

};



int main(int argc, char** argv)
{
    bool b0 = false ; 
    bool b1 = true ; 
    float f = 1.123f ; 
    double d = 1.123 ; 
    int i = -5 ;
    unsigned u = 6 ;

    char c = -128 ; 
    unsigned char uc = 255 ; 
    const char* cc = "hello const char*" ; 
    std::string s = "hello std::string" ; 


    Obj o ; 


    std::vector<float> v ; 
    v.push_back(101.101f) ; 

    dump(b0, b1,f,d,i,u, v);

 
    return 0 ;  // (*lldb*) Exit
}

