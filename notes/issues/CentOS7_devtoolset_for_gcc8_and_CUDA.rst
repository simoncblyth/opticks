CentOS7_devtoolset_for_gcc8_and_CUDA
======================================


* :google:`centos 7 devtoolset and CUDA`

* https://forums.developer.nvidia.com/t/rhel-centos-7-5-with-devtoolset-7-gcc-v-7-3-1-and-cuda-toolkit-v-10-0-130-compile-issue/68004



* https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#post-installation-actions




CentOS7 default gcc
    [blyth@localhost ~]$ gcc --version
    gcc (GCC) 4.8.5 20150623 (Red Hat 4.8.5-39)

CentOS7+devtoolset-7   
   [simon@localhost ~]$ gcc --version
   gcc (GCC) 7.3.1 20180303 (Red Hat 7.3.1-5)

CentOS7+devtoolset-8   
    [simon@localhost ~]$ gcc --version
    gcc (GCC) 8.3.1 20190311 (Red Hat 8.3.1-3)
    
CentOS7+devtoolset-9 
    [simon@localhost CLHEP.build]$ gcc --version
    gcc (GCC) 9.3.1 20200408 (Red Hat 9.3.1-2)



CentOS7 Default gcc 4.8.5, CUDA 10.1 nvcc 
--------------------------------------------

::

    [blyth@localhost opticks]$ thrap tests
    [blyth@localhost tests]$ nvcc rng.cu -o /tmp/rng 
    /usr/bin/ld: cannot open output file /tmp/rng: Permission denied
    collect2: error: ld returned 1 exit status
    [blyth@localhost tests]$ nvcc rng.cu -o /tmp/blyth/rng 
    [blyth@localhost tests]$ nvcc --version
    nvcc: NVIDIA (R) Cuda compiler driver
    Copyright (c) 2005-2019 NVIDIA Corporation
    Built on Fri_Feb__8_19:08:17_PST_2019
    Cuda compilation tools, release 10.1, V10.1.105
    [blyth@localhost tests]$ gcc --version
    gcc (GCC) 4.8.5 20150623 (Red Hat 4.8.5-39)
    Copyright (C) 2015 Free Software Foundation, Inc.
    This is free software; see the source for copying conditions.  There is NO
    warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.




CentOS7 Devtoolset-7 gcc 7.3.1, CUDA 10.1 nvcc 
------------------------------------------------


::

    [simon@localhost tests]$ nvcc rng.cu -o /tmp/rng
    /opt/rh/devtoolset-7/root/usr/include/c++/7/bits/basic_string.tcc: In instantiation of ‘static std::basic_string<_CharT, _Traits, _Alloc>::_Rep* std::basic_string<_CharT, _Traits, _Alloc>::_Rep::_S_create(std::basic_string<_CharT, _Traits, _Alloc>::size_type, std::basic_string<_CharT, _Traits, _Alloc>::size_type, const _Alloc&) [with _CharT = char16_t; _Traits = std::char_traits<char16_t>; _Alloc = std::allocator<char16_t>; std::basic_string<_CharT, _Traits, _Alloc>::size_type = long unsigned int]’:
    /opt/rh/devtoolset-7/root/usr/include/c++/7/bits/basic_string.tcc:578:28:   required from ‘static _CharT* std::basic_string<_CharT, _Traits, _Alloc>::_S_construct(_InIterator, _InIterator, const _Alloc&, std::forward_iterator_tag) [with _FwdIterator = const char16_t*; _CharT = char16_t; _Traits = std::char_traits<char16_t>; _Alloc = std::allocator<char16_t>]’
    /opt/rh/devtoolset-7/root/usr/include/c++/7/bits/basic_string.h:5033:20:   required from ‘static _CharT* std::basic_string<_CharT, _Traits, _Alloc>::_S_construct_aux(_InIterator, _InIterator, const _Alloc&, std::__false_type) [with _InIterator = const char16_t*; _CharT = char16_t; _Traits = std::char_traits<char16_t>; _Alloc = std::allocator<char16_t>]’
    /opt/rh/devtoolset-7/root/usr/include/c++/7/bits/basic_string.h:5054:24:   required from ‘static _CharT* std::basic_string<_CharT, _Traits, _Alloc>::_S_construct(_InIterator, _InIterator, const _Alloc&) [with _InIterator = const char16_t*; _CharT = char16_t; _Traits = std::char_traits<char16_t>; _Alloc = std::allocator<char16_t>]’
    /opt/rh/devtoolset-7/root/usr/include/c++/7/bits/basic_string.tcc:656:134:   required from ‘std::basic_string<_CharT, _Traits, _Alloc>::basic_string(const _CharT*, std::basic_string<_CharT, _Traits, _Alloc>::size_type, const _Alloc&) [with _CharT = char16_t; _Traits = std::char_traits<char16_t>; _Alloc = std::allocator<char16_t>; std::basic_string<_CharT, _Traits, _Alloc>::size_type = long unsigned int]’
    /opt/rh/devtoolset-7/root/usr/include/c++/7/bits/basic_string.h:6676:95:   required from here
    /opt/rh/devtoolset-7/root/usr/include/c++/7/bits/basic_string.tcc:1067:16: error: cannot call member function ‘void std::basic_string<_CharT, _Traits, _Alloc>::_Rep::_M_set_sharable() [with _CharT = char16_t; _Traits = std::char_traits<char16_t>; _Alloc = std::allocator<char16_t>]’ without object
           __p->_M_set_sharable();
           ~~~~~~~~~^~
    /opt/rh/devtoolset-7/root/usr/include/c++/7/bits/basic_string.tcc: In instantiation of ‘static std::basic_string<_CharT, _Traits, _Alloc>::_Rep* std::basic_string<_CharT, _Traits, _Alloc>::_Rep::_S_create(std::basic_string<_CharT, _Traits, _Alloc>::size_type, std::basic_string<_CharT, _Traits, _Alloc>::size_type, const _Alloc&) [with _CharT = char32_t; _Traits = std::char_traits<char32_t>; _Alloc = std::allocator<char32_t>; std::basic_string<_CharT, _Traits, _Alloc>::size_type = long unsigned int]’:
    /opt/rh/devtoolset-7/root/usr/include/c++/7/bits/basic_string.tcc:578:28:   required from ‘static _CharT* std::basic_string<_CharT, _Traits, _Alloc>::_S_construct(_InIterator, _InIterator, const _Alloc&, std::forward_iterator_tag) [with _FwdIterator = const char32_t*; _CharT = char32_t; _Traits = std::char_traits<char32_t>; _Alloc = std::allocator<char32_t>]’
    /opt/rh/devtoolset-7/root/usr/include/c++/7/bits/basic_string.h:5033:20:   required from ‘static _CharT* std::basic_string<_CharT, _Traits, _Alloc>::_S_construct_aux(_InIterator, _InIterator, const _Alloc&, std::__false_type) [with _InIterator = const char32_t*; _CharT = char32_t; _Traits = std::char_traits<char32_t>; _Alloc = std::allocator<char32_t>]’
    /opt/rh/devtoolset-7/root/usr/include/c++/7/bits/basic_string.h:5054:24:   required from ‘static _CharT* std::basic_string<_CharT, _Traits, _Alloc>::_S_construct(_InIterator, _InIterator, const _Alloc&) [with _InIterator = const char32_t*; _CharT = char32_t; _Traits = std::char_traits<char32_t>; _Alloc = std::allocator<char32_t>]’
    /opt/rh/devtoolset-7/root/usr/include/c++/7/bits/basic_string.tcc:656:134:   required from ‘std::basic_string<_CharT, _Traits, _Alloc>::basic_string(const _CharT*, std::basic_string<_CharT, _Traits, _Alloc>::size_type, const _Alloc&) [with _CharT = char32_t; _Traits = std::char_traits<char32_t>; _Alloc = std::allocator<char32_t>; std::basic_string<_CharT, _Traits, _Alloc>::size_type = long unsigned int]’
    /opt/rh/devtoolset-7/root/usr/include/c++/7/bits/basic_string.h:6681:95:   required from here
    /opt/rh/devtoolset-7/root/usr/include/c++/7/bits/basic_string.tcc:1067:16: error: cannot call member function ‘void std::basic_string<_CharT, _Traits, _Alloc>::_Rep::_M_set_sharable() [with _CharT = char32_t; _Traits = std::char_traits<char32_t>; _Alloc = std::allocator<char32_t>]’ without object



Curiously need the -std=c++11 flag with the different gcc
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


::

    [simon@localhost tests]$ nvcc rng.cu -std=c++11 -o /tmp/rng


    [simon@localhost tests]$ gcc --version
    gcc (GCC) 7.3.1 20180303 (Red Hat 7.3.1-5)
    Copyright (C) 2017 Free Software Foundation, Inc.
    This is free software; see the source for copying conditions.  There is NO
    warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.

    [simon@localhost tests]$ nvcc --version
    nvcc: NVIDIA (R) Cuda compiler driver
    Copyright (c) 2005-2019 NVIDIA Corporation
    Built on Fri_Feb__8_19:08:17_PST_2019
    Cuda compilation tools, release 10.1, V10.1.105
    [simon@localhost tests]$ t 




