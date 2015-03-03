# === func-gen- : cuda/texture/cudatex fgp cuda/texture/cudatex.bash fgn cudatex fgh cuda/texture
cudatex-src(){      echo cuda/texture/cudatex.bash ; }
cudatex-source(){   echo ${BASH_SOURCE:-$(env-home)/$(cudatex-src)} ; }
cudatex-vi(){       vi $(cudatex-source) ; }
cudatex-usage(){ cat << EOU

CUDATEX
========


Ref
----

* http://cuda-programming.blogspot.tw/2013/04/texture-references-object-in-cuda.html

Next Steps
-----------

* arrange to wavelength dependent props handling in ggeo- so have some
  real numbers to experiment with 


Wavelength dependent material/surface property lookups via texture objects ?
-------------------------------------------------------------------------------

* one dimensional, as interpolating off single qty: wavelength
* hardware linear interpolation 
* values can be float4 (double4 ?) 

  * pack four properties into one texture, maybe single
    texture for each material and surface would be sufficient

Compare with Chroma photon.h:fill_state::

    223 
    224     s.refractive_index1 = interp_property(material1, p.wavelength,
    225                                           material1->refractive_index);
    226     s.refractive_index2 = interp_property(material2, p.wavelength,
    227                                           material2->refractive_index);
    228     s.absorption_length = interp_property(material1, p.wavelength,
    229                                           material1->absorption_length);
    230     s.scattering_length = interp_property(material1, p.wavelength,
    231                                           material1->scattering_length);
    232     s.reemission_prob = interp_property(material1, p.wavelength,
    233                                         material1->reemission_prob);
    234 

    (chroma_env)delta:cuda blyth$ grep -l interp_property *.cu *.h
    cerenkov.h
    geometry.h
    photon.h

::

    741 __device__ int
    742 propagate_at_surface(Photon &p, State &s, curandState &rng, Geometry *geometry,
    743                      bool use_weights=false)
    744 {
    745     Surface *surface = geometry->surfaces[s.surface_index];
    746 
    747     if (surface->model == SURFACE_COMPLEX)
    748         return propagate_complex(p, s, rng, surface, use_weights);
    749     else if (surface->model == SURFACE_WLS)
    750         return propagate_at_wls(p, s, rng, surface, use_weights);
    751     else
    752     {
    753         // use default surface model: do a combination of specular and
    754         // diffuse reflection, detection, and absorption based on relative
    755         // probabilties
    756 
    757         // since the surface properties are interpolated linearly, we are
    758         // guaranteed that they still sum to 1.0.
    759         float detect = interp_property(surface, p.wavelength, surface->detect);
    760         float absorb = interp_property(surface, p.wavelength, surface->absorb);
    761         float reflect_diffuse = interp_property(surface, p.wavelength, surface->reflect_diffuse);
    762         float reflect_specular = interp_property(surface, p.wavelength, surface->reflect_specular);
    763 








EOU
}
cudatex-bdir(){ echo $(local-base)/env/cuda/texture ; }
cudatex-sdir(){ echo $(env-home)/cuda/texture ; }
cudatex-scd(){  cd $(cudatex-sdir); }
cudatex-bcd(){  cd $(cudatex-bdir); }
cudatex-cd(){   cudatex-scd ; }

cudatex-env(){      
   elocal- 
   cuda-
}


cudatex-name(){
  case $1 in 
    cudatex-tt) echo cuda_texture_test ;;
    cudatex-to) echo cuda_texture_object ;;
  esac
}


cudatex-tt(){ cudatex-- $FUNCNAME ; }
cudatex-to(){ cudatex-- $FUNCNAME ; }

cudatex-options(){ cat << EOO
-arch=sm_30
EOO
}

cudatex--(){
   local fn=${1:-cudatex-tt}
   local name=$(cudatex-name $fn)

   mkdir -p $(cudatex-bdir)
   local cmd="nvcc -o $(cudatex-bdir)/$name $(cudatex-options)  $(cudatex-sdir)/$name.cu"
   echo $msg $cmd
   eval $cmd

   cmd="$(cudatex-bdir)/$name"

   echo $msg $cmd
   eval $cmd
 
}
