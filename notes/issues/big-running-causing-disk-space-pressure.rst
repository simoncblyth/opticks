big-running-causing-disk-space-pressure
=============================================


Context
----------

* :doc:`30M-interop-launch-CUDA-invalid-address`


What to try 
---------------

* considered save slice used by NPY, to cut the array size at source : possible 
  but will take significat development 
 
* as half the arrays are coming from GPU cannot just stop writing to them beyond a certain size

* added --nosave option that trumps --save but thats a blunt approach as still need to save metadata

* actually best is to *revive the production running mode*, which 
  does not even collect the big arrays in the first place : only hits are downloaded 



ISSUE : large event running space problem
----------------------------------------------

::

    [blyth@localhost torch]$ du -hs -- *
    4.0K    0
    106M    1
    105M    -1
    3.3G    10
    3.3G    -10
    1.3G    100
    1.3G    -100
    6.5G    20
    665M    200
    665M    -200
    5.3G    30
    4.0K    DeltaTime.ini
    4.0K    DeltaVM.ini
    16K Opticks.npy
    4.0K    OpticksProfileAccLabels.npy
    4.0K    OpticksProfileAcc.npy
    8.0K    OpticksProfileLabels.npy
    4.0K    OpticksProfile.npy
    4.0K    Time.ini
    4.0K    VM.ini
    [blyth@localhost torch]$ pwd
    /tmp/blyth/opticks/tboolean-box/evt/tboolean-box/torch
    [blyth@localhost torch]$ 



::

    home/blyth/anaconda2/lib/python2.7/site-packages/numpy/core/memmap.pyc in __new__(subtype, filename, dtype, mode, offset, shape, order)
        262         bytes -= start
        263         array_offset = offset - start
    --> 264         mm = mmap.mmap(fid.fileno(), bytes, access=acc, offset=start)
        265 
        266         self = ndarray.__new__(subtype, shape, dtype=descr, buffer=mm,

    ValueError: mmap length is greater than file size
    > /home/blyth/anaconda2/lib/python2.7/site-packages/numpy/core/memmap.py(264)__new__()
        262         bytes -= start
        263         array_offset = offset - start
    --> 264         mm = mmap.mmap(fid.fileno(), bytes, access=acc, offset=start)
        265 
        266         self = ndarray.__new__(subtype, shape, dtype=descr, buffer=mm,

    ipdb> u
    > /home/blyth/anaconda2/lib/python2.7/site-packages/numpy/lib/format.py(802)open_memmap()
        800 
        801     marray = numpy.memmap(filename, dtype=dtype, shape=shape, order=order,
    --> 802         mode=mode, offset=offset)
        803 
        804     return marray

    ipdb> u
    > /home/blyth/anaconda2/lib/python2.7/site-packages/numpy/lib/npyio.py(434)load()
        432     finally:
        433         if own_fid:
    --> 434             fid.close()
        435 
        436 

    ipdb> u
    > /home/blyth/opticks/ana/nload.py(255)load_()
        253                 arr = np.load(path)
        254             else:
    --> 255                 arr = np.load(path, mmap_mode="r")
        256                 oshape = arr.shape        #
        257                 arr = arr[msli]

    ipdb> p path
    '/tmp/blyth/opticks/tboolean-box/evt/tboolean-box/torch/30/rx.npy'

