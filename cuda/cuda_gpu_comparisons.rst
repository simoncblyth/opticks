CUDA GPU Comparisons
======================

Multi GPU scaling
-----------------

I see there are complications
to getting pycuda to work with multiple GPUs:

* http://wiki.tiker.net/PyCuda/FrequentlyAskedQuestions#How_about_multiple_GPUs.3F
* http://wiki.tiker.net/PyCuda/FrequentlyAskedQuestions#threading

It may be more convenient to call the Chroma CUDA  C kernels 
directly in C/C++ rather than using it via PyCUDA ?

Will performance scale with GPUs, eg for a 
workstation with 4 NVIDIA Tesla K20m::

    4*5  ~ 20 in GFLOPS
    4*8 ~  32 in cores



NVIDIA GeForce GT 750M  vs NVIDIA Tesla K20m
-----------------------------------------------

Comparing NVIDIA GeForce GT 750M  vs NVIDIA Tesla K20m


* 3524./711 = 4.95   ## factor ~5 in GFLOPS
* 2946./384. = 7.67  ## factor ~8 in cores

::

    In [1]: 3524./711 = 4.95     ## factor ~5 in GFLOPS
    Out[1]: 4.956399437412095

    In [2]: 2946./384. = 7.67    ## factor ~8 in cores
    Out[2]: 7.671875



NVIDIA GeForce GT 750M
-----------------------

* http://www.techpowerup.com/gpudb/2527/geforce-gt-750m-mac-edition.html


The GeForce GT 750M Mac Edition is a graphics card by NVIDIA, launched in November 2013. 
Built on the 28 nm process, and based on the GK107 graphics
processor, in its N14P-GT variant, the card supports DirectX 11.0. 
The GK107 graphics processor is an average sized chip 
with a die area of 118 mm^2 and 1,270 million transistors. 

It features 384 shading units, 32 texture mapping units and 16 ROPs. 
NVIDIA has placed 2,048 MB GDDR5 memory on the card, which are connected 
using a 128-bit memory interface. 
The GPU is operating at a frequency of 926 MHz, memory is running at 1254 MHz.  
We recommend the NVIDIA GeForce GT 750M Mac Edition for gaming 
with highest details at resolutions up to, and including, 1280x720.  
Being a mxm module card, its power draw is rated
at 50 W maximum. GeForce GT 750M Mac Edition is connected to the rest of the
system using a PCIe 3.0 x16 interface.

::

    Compute Capability: CUDA 3.0

    Shading Units:  384

    TMUs:   32

    ROPs:   16

    SMX Count:  2

    Pixel Rate: 7.41 GPixel/s

    Texture Rate:   29.6 GTexel/s

    Floating-point performance: 711.2 GFLOPS




NVIDIA Tesla K20m
-------------------


* http://www.techpowerup.com/gpudb/2029/tesla-k20m.html

The Tesla K20m is a high-end professional graphics card by NVIDIA, launched in January 2013. 
Built on the 28 nm process, and based on the GK110 graphics
processor, the card supports DirectX 11.0. 
The GK110 graphics processor is a large chip with a die area of 561 mm^2 and 7,080 million transistors. 
Unlike the fully unlocked GeForce GTX TITAN, which uses the same GPU but has all 2688
shaders enabled, NVIDIA has disabled some shading units on the Tesla K20m to
reach the product's target shader count. 

It features 2496 shading units, 208 texture mapping units and 40 ROPs. 
NVIDIA has placed 5,120 MB GDDR5 memory on the card, which are connected 
using a 320-bit memory interface. 
The GPU is operating at a frequency of 706 MHz, memory is running at 1300 MHz.  
We recommend the NVIDIA Tesla K20m for gaming with highest details at resolutions
up to, and including, 1920x1080.  
Being a dual-slot card, its power draw is rated at 225 W maximum. 
Tesla K20m is connected to the rest of the system using a PCIe 2.0 x16 interface. 
The card measures 267 mm in length, and features a dual-slot cooling solution. 
Its price at launch was 3,199 US Dollars.


::

    Compute Capability: CUDA 3.5

    Shading Units:  2496

    TMUs:   208

    ROPs:   40

    SMX Count:  13

    Pixel Rate: 36.7 GPixel/s

    Texture Rate:   147 GTexel/s

    Floating-point performance: 3,524 GFLOPS





