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



