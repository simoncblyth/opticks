max_photon_launch_size_with_8GB_VRAM
=======================================


Simple rule of thumb from qudarap/QCurandState.hh seems to be over optimistic
--------------------------------------------------------------------------------

Decide on max size of photon launches by scaling from 48G for 400M, eg with 8G VRAM::

    In [2]: 8.*400./48.
    Out[2]: 66.66666666666667    ## so you might aim for 60M photons max if only hand 8G VRAM


When there is less available VRAM, need a more precise rule of thumb. 



Zike Reports
----------------

For the limit from my VRAM of 8G, it's not 60M. To set this number to 60M, I generate a 61M QCurandState file, and load it. 
However, a fault came out::

    [neT-Log] Begin to Simulate 45000010 in this launch.

    2022-11-15 15:16:16.252 ERROR [2898696] [QU::_cudaMalloc@212] save salloc record to /tmp/sblinux/opticks/GEOM/neutrinoT

    terminate called after throwing an instance of 'QUDA_Exception'
      what():  CUDA call (max_photon*sizeof(sphoton) ) failed with error: 'out of memory' (/home/sblinux/opticks/qudarap/QU.cc:206)

    salloc::desc alloc.size 5 label.size 5

         [           size   num_items sizeof_item       spare]    size_GB    percent label
         [        (bytes)                                    ]   size/1e9            
         [            208           1         208           0]       0.00       0.00 QEvent::QEvent/sevent
         [        8294400     2073600           4           0]       0.01       0.20 Frame::DeviceAllo:num_pixels
         [       96000000     1000000          96           0]       0.10       2.26 device_alloc_genstep:quad6
         [      244000000    61000000           4           0]       0.24       5.74 device_alloc_genstep:int seed
         [     3904000000    61000000          64           0]       3.90      91.81 max_photon*sizeof(sphoton)

     tot       4252294608                                            4.25





