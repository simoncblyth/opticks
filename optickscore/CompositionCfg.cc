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

#include "NGLM.hpp"
#include "Composition.hh"
#include "CompositionCfg.hh"



template OKCORE_API void BCfg::addOptionF<Composition>(Composition*, const char*, const char* );
template OKCORE_API void BCfg::addOptionI<Composition>(Composition*, const char*, const char* );
template OKCORE_API void BCfg::addOptionS<Composition>(Composition*, const char*, const char* );





template <class Listener>
CompositionCfg<Listener>::CompositionCfg(const char* name, Listener* listener, bool live) 
    : 
    BCfg(name, live) 
{
       addOptionI<Listener>(listener, Listener::PRINT,    "Print");
       addOptionS<Listener>(listener, Listener::SELECT,   "Selection, four comma delimited integers");

       addOptionS<Listener>(listener, Listener::EYEW,    "Three Comma delimited world space eye coordinates, eg 300,300,0 " );
       addOptionS<Listener>(listener, Listener::LOOKW,   "Three Comma delimited world space look coordinates, eg 0,0,300 " );


       addOptionS<Listener>(listener, Listener::PICKPHOTON, 
           "[UDP only], up to 4 comma delimited integers, eg:\n"
           "10000   : target view at the center extent \n" 
           "10000,1 : as above but hide other records \n" 
           "\n"
           "see CompositionCfg.hh\n"
      );

}





template class OKCORE_API CompositionCfg<Composition> ;

