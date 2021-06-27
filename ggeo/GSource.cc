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

#include <cassert>

#include "GAry.hh"
#include "GDomain.hh"
#include "GProperty.hh"
#include "GPropertyMap.hh"
#include "GPropertyLib.hh"
#include "GSource.hh"


GSource::GSource(GSource* other) : GPropertyMap<double>(other)
{
}

GSource::GSource(const char* name, unsigned int index) : GPropertyMap<double>(name, index, "source")
{
   init();
}

GSource::~GSource()
{
}

void GSource::Summary(const char* msg )
{
    GPropertyMap<double>::Summary(msg);
}


void GSource::init()
{
    GDomain<double>* sd = GPropertyLib::getDefaultDomain();
    setStandardDomain(sd);
}


GSource* GSource::make_blackbody_source(const char* name, unsigned int index, double /*kelvin*/)
{
    GSource* source = new GSource(name, index);

    GProperty<double>* radiance = GProperty<double>::planck_spectral_radiance( source->getStandardDomain(), 6500. );

    assert(radiance) ;

    source->addProperty("radiance", radiance );

    return source ; 
}

