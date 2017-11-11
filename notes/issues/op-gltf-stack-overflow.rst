op-gltf-stack-overflow
==========================

Full DYB gltf geom running is giving stack overflow for some photons...

* didnt see this is a very long time ? What changed ?

* seems to be only from the propagation, not the ray trace 


::

    op --gltf 1
    op --gltf 3

    2017-11-11 14:02:11.863 INFO  [4410792] [OContext::launch@307] OContext::launch PRELAUNCH DONE
    2017-11-11 14:02:11.864 INFO  [4410792] [OContext::launch@309] OContext::launch PRELAUNCH time: 0.365464
    2017-11-11 14:02:11.864 INFO  [4410792] [OPropagator::prelaunch@160] 1 : (0;100000,1) prelaunch_times vali,comp,prel,lnch  0.0048 8.4512 0.3655 0.0000
    Caught RT_EXCEPTION_STACK_OVERFLOW
      launch index : 1792, 0, 0
    Caught RT_EXCEPTION_STACK_OVERFLOW
      launch index : 1793, 0, 0
    Caught RT_EXCEPTION_STACK_OVERFLOW
      launch index : 1796, 0, 0
    Caught RT_EXCEPTION_STACK_OVERFLOW
      launch index : 1798, 0, 0
    Caught RT_EXCEPTION_STACK_OVERFLOW
      launch index : 1814, 0, 0
    Caught RT_EXCEPTION_STACK_OVERFLOW
      launch index : 1800, 0, 0
    Caught RT_EXCEPTION_STACK_OVERFLOW
      launch index : 1802, 0, 0
    Caught RT_EXCEPTION_STACK_OVERFLOW

