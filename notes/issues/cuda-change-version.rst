FIXED : cuda-change-version reluctant : was due to om-clean was not cleaning 
===============================================================================

Trying to test CUDA 10.1 starting from an CUDA 9.2 installation.

Even with the nuclear option::

   cd ~/opticks
   om-clean
   om-conf

Still there are 9.2 appearing. Where are they coming from ? Loadsa them::

   find ~/local/opticks/build -name '*.cmake' -exec grep -H cuda {} \;


~/opticks/examples/UseCUDA UseUseCUDA
------------------------------------------

* succeeds to find 10.1 and ignore 9.2 just by ensuring clean PATH and LD_LIBRARY_PATH environment
  with only one SDK in the dirs 

* these ones worked without om-clean as the go.sh did the cleaning 


CUDA libs : notice one installed with display driver into /usr/lib too
-------------------------------------------------------------------------------

::

    [blyth@localhost ~]$ ll /usr/lib/*cuda*
    -rwxr-xr-x. 1 root root 14787552 Apr  9 11:44 /usr/lib/libcuda.so.418.56
    lrwxrwxrwx. 1 root root       17 Apr  9 11:44 /usr/lib/libcuda.so.1 -> libcuda.so.418.56
    lrwxrwxrwx. 1 root root       12 Apr  9 11:44 /usr/lib/libcuda.so -> libcuda.so.1

    [blyth@localhost ~]$ ll /usr/local/cuda-10.1/lib64/*cuda*
    lrwxrwxrwx. 1 root root     21 Apr  8 14:43 /usr/local/cuda-10.1/lib64/libcudart.so.10.1 -> libcudart.so.10.1.105
    -rwxr-xr-x. 1 root root 504480 Apr  8 14:43 /usr/local/cuda-10.1/lib64/libcudart.so.10.1.105
    -rw-r--r--. 1 root root 888488 Apr  8 14:43 /usr/local/cuda-10.1/lib64/libcudart_static.a
    -rw-r--r--. 1 root root 717772 Apr  8 14:43 /usr/local/cuda-10.1/lib64/libcudadevrt.a
    lrwxrwxrwx. 1 root root     17 Apr  8 14:43 /usr/local/cuda-10.1/lib64/libcudart.so -> libcudart.so.10.1

    [blyth@localhost ~]$ ll /usr/local/cuda-9.2/lib64/*cuda*
    lrwxrwxrwx. 1 root root     16 Jul  5  2018 /usr/local/cuda-9.2/lib64/libcudart.so -> libcudart.so.9.2
    lrwxrwxrwx. 1 root root     19 Jul  5  2018 /usr/local/cuda-9.2/lib64/libcudart.so.9.2 -> libcudart.so.9.2.88
    -rwxr-xr-x. 1 root root 430616 Jul  5  2018 /usr/local/cuda-9.2/lib64/libcudart.so.9.2.88
    -rw-r--r--. 1 root root 621252 Jul  5  2018 /usr/local/cuda-9.2/lib64/libcudadevrt.a
    -rw-r--r--. 1 root root 815552 Jul  5  2018 /usr/local/cuda-9.2/lib64/libcudart_static.a
    [blyth@localhost ~]$ 



 
