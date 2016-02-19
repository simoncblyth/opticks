Box Asymmetry
===================

PmtInBox shows almost no scatters in two directions, whereas expecting symmetry::

   ggv-pmt-test

* Switching from analytic box to analytic sphere shows no asymmetry
* Switching off analytic also shows no asymmetry, but currently the
  analytic applies to the PMT too (which is too unrealistic to pursue)  


Retreat to single box, can see the scatter asymmetry by flipping the
analytic switch::

    ggv-box-test(){
       type $FUNCNAME

       local torch_config=(
                     type=disclin
                     photons=500000
                     wavelength=380 
                     frame=1
                     source=0,0,300
                     target=0,0,0
                     radius=100
                     zenithazimuth=0,1,0,1
                     material=Vacuum
                   )    

       local test_config=(
                     mode=BoxInBox
                     shape=box
                     analytic=1
                     boundary=Rock//perfectAbsorbSurface/MineralOil
                     parameters=0,0,0,300
                       )    

       ggv \
           --test --testconfig "$(join _ ${test_config[@]})" \
           --torch --torchconfig "$(join _ ${torch_config[@]})" \
           --animtimemax 10 \ 
           --cat BoxInBox \
           --eye 0.5,0.5,0.0 \
           $*   

    }


Issue apparent for "TO SC SA" but thats probably coincidental and its just a geometry issue ?

Fixed issue with *intersect_box* was wrongly to conflating the slightly enlarged aabb 
with the contained box.




