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

// TEST=OpticksProfileTest om-t

#include "OpticksProfile.hh"
#include "NGLM.hpp"
#include "OPTICKS_LOG.hh"



struct Leak
{
    static unsigned count ; 

    unsigned n ; 
    unsigned i ; 
    char*    d ;
 
    Leak(unsigned n_) 
        :
        n(n_),
        i(count),
        d(new char[n])
    {
        count += 1 ;   
    } 

    std::string desc() const
    {
        std::stringstream ss ; 
        ss 
            << " i " << i 
            << " n " << n
            ;
        return ss.str();   
    }

    ~Leak()
    {
        LOG(info) << desc() ; 
        delete[] d ; 
    }

};


unsigned Leak::count = 0 ; 




struct OpticksProfileTest 
{
    OpticksProfile* op ;
    unsigned long long KB ;  
    unsigned long long MB ;  
    unsigned acc ; 
    unsigned tagoffset ; 

    std::vector<Leak*>  m_leak ; 
 

    OpticksProfileTest() 
        : 
        op(new OpticksProfile),
        //KB(1024),
        KB(1000),
        MB(KB*KB), 
        acc(op->accumulateAdd("OpticksProfileTest::check")),
        tagoffset(0)   
    {
        //unsigned long long MB2 = 1 << 10 << 10 ; 
        //assert( MB == MB2 );  

        op->setDir("$TMP/OpticksProfileTest") ; // canonically done from Opticks::configure 
        op->setStamp(true); 
    }


    void basics()
    {
        new char[MB] ; 
        op->stamp( "red:yellow:purple", tagoffset);
        new char[MB] ; 
        op->stamp( "green:pink:violet", tagoffset);
        new char[MB] ; 
        op->stamp( "blue:cyan:indigo", tagoffset);
        op->save(); 
    } 


    void testAccOne()
    {
         op->accumulateStart(acc); 
         new char[KB] ; 
         op->accumulateStop(acc); 
    }

    void testAccMany()
    {
         op->stamp("_loopcheck", tagoffset); 
         for(unsigned i=0 ; i < 100000 ; i++ ) testAccOne(); 
         op->stamp("loopcheck", tagoffset); 

         op->dump();
    }



    void testVecLeak()
    {
        op->stamp( "_testVecLeak", tagoffset);
          
        for(unsigned i=0 ; i < 200 ; i++)
        {
            m_leak.push_back(new Leak(MB)); 
        }


        for(unsigned i=0 ; i < m_leak.size() ; i++)
        {
             Leak* l = m_leak[i] ; 
             delete l ; 
        }
        m_leak.clear(); 


        op->stamp( "testVecLeak", tagoffset);
        op->dump();
    }




};







int main(int argc, char** argv)
{
    OPTICKS_LOG(argc,argv);
    
    LOG(info) << argv[0] ;

    OpticksProfileTest opt ;  


    //opt.basics();  
    //opt.testAccMany();  
    opt.testVecLeak();  



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


