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

#include "GMaterial.hh"
#include "GDomain.hh"
#include "GPropertyLib.hh"

GMaterial::GMaterial(GMaterial* other, GDomain<float>* domain ) 
    : 
    GPropertyMap<float>(other, domain)
{
}

GMaterial::GMaterial(const char* name, unsigned int index) 
    : 
    GPropertyMap<float>(name, index, "material")
{
    init();
}

void GMaterial::init()
{
    setStandardDomain( GDomain<float>::GetDefaultDomain()) ;   
}

GMaterial::~GMaterial()
{
}

void GMaterial::Summary(const char* msg )
{
    GPropertyMap<float>::Summary(msg);
}



