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

#pragma once

//#include <boost/random/mersenne_twister.hpp>
#include <boost/random/inversive_congruential.hpp>

#include <boost/random/uniform_int.hpp>
#include <boost/random/uniform_real.hpp>
#include <boost/random/variate_generator.hpp>

#include <glm/fwd.hpp>


#include "NBBox.hpp"

#include "NPY_API_EXPORT.hh"

class NPY_API NGenerator 
{
    //typedef boost::mt19937          RNG_t;
    typedef boost::hellekalek1995   RNG_t ; 
    typedef boost::uniform_real<>   Distrib_t;
    typedef boost::variate_generator< RNG_t, Distrib_t > Generator_t ;

    public:
        NGenerator(const nbbox& bb);
        void operator()(nvec3& xyz);
        void operator()(glm::vec3& xyz);
    private:
        nbbox m_bb ; 
        nvec3 m_side ; 
        RNG_t m_rng;
        Distrib_t m_dist ;
        Generator_t m_gen ; 
};



