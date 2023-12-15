thrust_system_system_error_cudaErrorInvalidDeviceFunction
=============================================================


Issue from Alexsey (CUDA 12 + Driver ) 
------------------------------------------------------------

::

    ndeed, I just checked out and compiled your latest version Hans.
    I attach the log of attempting to run it with:

    ./CaTS -g simpleLArTPC.gdml -m time.mac > error_cats.log

    I attach the error_cats.log to see what is happening during the execution.
    Also below is the screen output :
    --------------------------------------------------------------------------------------------
    opticks@wdeap1:~/test/CaTS-install/bin$ ./CaTS -g simpleLArTPC.gdml -m time.mac > error_cats.log
    SLOG::EnvLevel adjusting loglevel by envvar   key SEventConfig level INFO fallback DEBUG upper_level INFO
    SLOG::EnvLevel adjusting loglevel by envvar   key SEvt level INFO fallback DEBUG upper_level INFO
    SLOG::EnvLevel adjusting loglevel by envvar   key CSGOptiX level INFO fallback DEBUG upper_level INFO
    SLOG::EnvLevel adjusting loglevel by envvar   key G4CXOpticks level INFO fallback DEBUG upper_level INFO
    U4Tree::initRayleigh rayleigh_table
    U4PhysicsTable::desc proc YES
     procName OpRayleigh
     table YES
     tab (6, 101, 2, )

    stree::init_material_mapping level > 1 [2] desc_mt
    stree::desc_mt mtname 3 mtname_no_rindex 0 mtindex 3 mtline 3 mtindex.mn 0 mtindex.mx 4
     i   0 mtindex   4 mtline  11 mtname Glass
     i   1 mtindex   0 mtline   7 mtname liquidAr
     i   2 mtindex   2 mtline   0 mtname Air

    [ U4Tree::identifySensitive
    [ U4Tree::identifySensitiveInstances num_factor 0 st.sensor_count 0
    ] U4Tree::identifySensitiveInstances num_factor 0 st.sensor_count 0
    [ U4Tree::identifySensitiveGlobals st.sensor_count 0 remainder.size 7
    U4Tree::identifySensitiveGlobals i       0 nidx      0 sensor_id      -1 sensor_index      -1 pvn World_PV ppvn -
    U4Tree::identifySensitiveGlobals i       1 nidx      1 sensor_id      -1 sensor_index      -1 pvn Obj ppvn World_PV
    U4Tree::identifySensitiveGlobals i       2 nidx      2 sensor_id      -1 sensor_index      -1 pvn Det2 ppvn Obj
    U4Tree::identifySensitiveGlobals i       3 nidx      3 sensor_id      -1 sensor_index      -1 pvn Det3 ppvn Obj
    U4Tree::identifySensitiveGlobals i       4 nidx      4 sensor_id      -1 sensor_index      -1 pvn Det0 ppvn Obj
    U4Tree::identifySensitiveGlobals i       5 nidx      5 sensor_id      -1 sensor_index      -1 pvn Det4 ppvn Obj
    U4Tree::identifySensitiveGlobals i       6 nidx      6 sensor_id      -1 sensor_index      -1 pvn Det1 ppvn Obj
    ] U4Tree::identifySensitiveGlobals  st.sensor_count 0 remainder.size 7
    ] U4Tree::identifySensitive st.sensor_count 0
    terminate called after throwing an instance of 'thrust::system::system_error'
      what():  after reduction step 1: cudaErrorInvalidDeviceFunction: invalid device function
    Aborted (core dumped)



    Do you have an idea of what is going on now?

    I tied using gdb to see more and I got the following at the end:


    2023-12-14 12:54:07.680 INFO  [2854071] [SEvt::addInputGenstep@716]
    terminate called after throwing an instance of 'thrust::system::system_error'
      what():  after reduction step 1: cudaErrorInvalidDeviceFunction: invalid device function

    Thread 1 "CaTS" received signal SIGABRT, Aborted.
    __GI_raise (sig=sig@entry=6) at ../sysdeps/unix/sysv/linux/raise.c:50
    50 ../sysdeps/unix/sysv/linux/raise.c: No such file or directory.

    Which leads me to believe that perhaps I should go to later optix package as Simon suggests.
    I will try that now...

    kindest regards,

    Aleksey




Grepping the thrust source::

    N[blyth@localhost thrust]$ pwd
    /usr/local/cuda/targets/x86_64-linux/include/thrust

    N[blyth@localhost thrust]$ find . -type f -exec grep -H "after reduction" {} \;
    ./system/cuda/detail/reduce.h:  cuda_cub::throw_on_error(status, "after reduction step 1");
    ./system/cuda/detail/reduce.h:  cuda_cub::throw_on_error(status, "after reduction step 2");
    ./system/cuda/detail/async/reduce.h:  , "after reduction sizing"
    ./system/cuda/detail/async/reduce.h:  , "after reduction launch"
    ./system/cuda/detail/async/reduce.h:  , "after reduction sizing"
    ./system/cuda/detail/async/reduce.h:  , "after reduction launch"
    N[blyth@localhost thrust]$ 


    0942   // Determine temporary device storage requirements.
     943 
     944   size_t tmp_size = 0;
     945 
     946   THRUST_INDEX_TYPE_DISPATCH2(status,
     947     cub::DeviceReduce::Reduce,
     948     (cub::DispatchReduce<
     949         InputIt, T*, Size, BinaryOp
     950     >::Dispatch),
     951     num_items,
     952     (NULL, tmp_size, first, reinterpret_cast<T*>(NULL),
     953         num_items_fixed, binary_op, init, stream,
     954         THRUST_DEBUG_SYNC_FLAG));
     955   cuda_cub::throw_on_error(status, "after reduction step 1");
     956 
     957   // Allocate temporary storage.
     958 

At a higher level this is probably coming from qudarap::

     478 /**
     479 QEvent::count_genstep_photons
     480 ------------------------------
     481 
     482 thrust::reduce using strided iterator summing over GPU side gensteps 
     483 
     484 **/
     485 
     486 extern "C" unsigned QEvent_count_genstep_photons(sevent* evt) ;
     487 unsigned QEvent::count_genstep_photons()
     488 {
     489    return QEvent_count_genstep_photons( evt );
     490 }


So I guess the stack is::

    QEvent::setGenstepUpload
    QEvent::count_genstep_photons_and_fill_seed_buffer    
    QEvent_count_genstep_photons    (from QEvent.cu)

That is using thrust::reduce to sum up the photons from the gensteps::

    084 extern "C" unsigned QEvent_count_genstep_photons(sevent* evt)
     85 {
     86     typedef typename thrust::device_vector<int>::iterator Iterator;
     87 
     88     thrust::device_ptr<int> t_gs = thrust::device_pointer_cast( (int*)evt->genstep ) ;
     89 
     90 #ifdef DEBUG_QEVENT
     91     printf("//QEvent_count_genstep_photons sevent::genstep_numphoton_offset %d  sevent::genstep_itemsize  %d  \n",
     92             sevent::genstep_numphoton_offset, sevent::genstep_itemsize );
     93 #endif
     94 
     95     strided_range<Iterator> gs_pho(
     96         t_gs + sevent::genstep_numphoton_offset,
     97         t_gs + evt->num_genstep*sevent::genstep_itemsize ,
     98         sevent::genstep_itemsize );    // begin, end, stride 
     99 
    100     evt->num_seed = thrust::reduce(gs_pho.begin(), gs_pho.end() );


And its doing it using standard thrust techniques. 
So check if QEventTest fails which is a standalone test of this::

    QEventTest 

Also check all tests::
 
    opticks-t 
    

Googling for the error yields:

   https://github.com/NVIDIA/thrust/issues/1737
   https://github.com/NVIDIA/thrust/issues/1401

Which suggests the error may be related to compilation options
in use regards symbol visibility. Or the use of multiple 
thrust versions.   Have you changed your CUDA version recently ?

I would also make sure to do a very clean build before doing 
anything else. 






 




Hi Hans, Alexsey, 

Good to see movement with your issue. 

The thrust errors Alexsey is getting makes be suspect 
mixing between CUDA versions or incompatibilities
between the CUDA and Driver versions.


From Log/opticks.txt I notice you 
are separately installing the driver and CUDA
and trusting the package manager to 
get the right combination of versions. 

With CUDA and Drivers it is not a good idea to just 
get the latest version, you need more control than that.


> making installation on POP OS 20.04
> 
> install nvidia drivers:
> 
> sudo apt install system76-driver-nvidia
> 
> check with nvidia-smi
> 
> 
> install CUDA:
> 
> sudo apt install system76-cuda-latest
> 
> reboot and check:
> 
> nvcc -V


I trust NVIDIA more that system package managers with regard 
to CUDA and Driver versions. 
This is particularly critical with OptiX 
because the implementation is all in the Driver, there
is no OptiX library anymore. 


I would suggest you use your package manager to 
remove the NVIDIA Driver and CUDA.
Also check for any other attempts to 
install CUDA and remove them from your system.

Then with a CUDA clean system use the 
runfile you downloaded earlier from NVIDIA 
to reinstall following:

    https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html

Its important to prepare ahead as you need to do the 
runfile install without the GUI using "init 3" or equivalent. 
The runfile nowadays includes both CUDA and the appropriate Driver.  
Hence you are certain to get an appropriate combination of versions. 


::

   U4Tree::identifySensitiveGlobals i       5 nidx      5 sensor_id      -1 sensor_index      -1 pvn Det4 ppvn Obj
    U4Tree::identifySensitiveGlobals i       6 nidx      6 sensor_id      -1 sensor_index      -1 pvn Det1 ppvn Obj
    ] U4Tree::identifySensitiveGlobals  st.sensor_count 0 remainder.size 7
    ] U4Tree::identifySensitive st.sensor_count 0
    terminate called after throwing an instance of 'thrust::system::system_error'
      what():  after reduction step 1: cudaErrorInvalidDeviceFunction: invalid device function
    Aborted (core dumped)


Grepping thrust source for "after reduction step" pinpoints the issue in::

   /usr/local/cuda/targets/x86_64-linux/include/thrust/system/cuda/detail/reduce.h


So I guess the stack is something like::

    QEvent::setGenstepUpload
    QEvent::count_genstep_photons_and_fill_seed_buffer    
    QEvent_count_genstep_photons    (from QEvent.cu)
    thrust::reduce 

You can test this code standalone using::

     QEventTest 

Also you might as well run all the tests::
 
    opticks-t 
    


> build cuda samples:
> cp -r /usr/lib/cuda-11.2/samples /home/opticks/
> cd /home/opticks/samples
> make --ignore-errors

Its good to test your install using the CUDA samples. 
Opticks does not need anything from them anymore. 


Googling for thrust error yields:

   https://github.com/NVIDIA/thrust/issues/1737
   https://github.com/NVIDIA/thrust/issues/1401

Which suggests the error may be related to compilation options
in use regards symbol visibility. Or the use of multiple
thrust versions.   Have you changed your CUDA version recently ?

But I suggest to avoid that rabbit hole, as other CUDA+Driver version 
combinations are not giving the issue, and test first with 
more standard version combinations. 


> change opticks/optickscore/OpticksSwitches.h
>
> so that:
>
> #define WITH_SKIPAHEAD 1

Optickscore is no longer an active package.




> I am observing a completely different crash see below. As one can see below
> (may be need to create protection when no photons need to be simulated.)
>

I have changed the behaviour of simulate when called with no gensteps to just 
giving a warning.  But until you update to a future tag you should
avoid calling simulate without any gensteps.


Simon










