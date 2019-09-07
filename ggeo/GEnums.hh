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

// GGEO_API
// documents wavelength_texture property ordering for line 0.5f 
// split into separate header to allow from kernel usage
//
enum {
   e_refractive_index,
   e_absorption_length,
   e_scattering_length,
   e_reemission_prob
};

enum {
   e_detect,
   e_absorb,
   e_reflect_specular,
   e_reflect_diffuse
};

   //e_reemission_cdf,
enum {
   e_extra_x,
   e_extra_y,
   e_extra_z,
   e_extra_w
};



