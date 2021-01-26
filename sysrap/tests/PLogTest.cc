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

// om-;TEST=PLogTest om-t
#include <plog/Log.h>

#include <plog/Formatters/MessageOnlyFormatter.h>
#include <plog/Formatters/FuncMessageFormatter.h>
#include <plog/Formatters/TxtFormatter.h>
#include <plog/Formatters/CsvFormatter.h>

#include <plog/Appenders/ColorConsoleAppender.h>

// translate from boost log levels to plog  ... but this are dangerous
// Better to do using plog::verbose etc...

#define fatal plog::fatal
#define error plog::error
#define warning plog::warning
#define info plog::info
#define debug plog::debug
#define trace plog::verbose

using namespace plog ; 




int main(int, char** argv)
{
    //typedef plog::MessageOnlyFormatter FMT ; 
    typedef plog::FuncMessageFormatter FMT ;     // useful for log comparisons
    //typedef plog::TxtFormatter         FMT ;   // this the default full format 
    //typedef plog::CsvFormatter         FMT ; 

    static plog::ColorConsoleAppender<FMT> consoleAppender;
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


    int ilevel = info ; 
    plog::Severity level = info ; 

    LOG(level) << "gello " ; 
    LOG((plog::Severity)ilevel) << "i-gello " ; 

    std::cout << " (int)fatal   " << (int)fatal << std::endl ;  
    std::cout << " (int)error   " << (int)error << std::endl ;  
    std::cout << " (int)warning " << (int)warning << std::endl ;  
    std::cout << " (int)info    " << (int)info << std::endl ;  
    std::cout << " (int)debug   " << (int)debug << std::endl ;  
    std::cout << " (int)verbose " << (int)verbose << std::endl ;  




    return 0 ; 
}

// om-;TEST=PLogTest om-t 

