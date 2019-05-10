profile-save-ptree-is-too-deep
=================================


FIXED : Reproduced in optickscore/tests/OpticksProfileTest.cc SOLUTION : dont use dots in labels
------------------------------------------------------------------------------------------------------

* labels with dots are interpreted as heirarchy levels by boost property_tree : so dont do that


Issuse
----------


::

    geocache-;geocache-j1808-v4
    ...
    2019-05-10 15:34:05.426 INFO  [397182] [BFile::preparePath@610] created directory /home/blyth/local/opticks/geocache/OKX4Test_lWorld0x4bc2710_PV_g4live/g4ok_gltf/699463ea0065185a7ffaf10d4935fc61/1/source/evt/g4live/torch
    terminate called after throwing an instance of 'boost::exception_detail::clone_impl<boost::exception_detail::error_info_injector<boost::property_tree::ini_parser::ini_parser_error> >'
      what():  /home/blyth/local/opticks/geocache/OKX4Test_lWorld0x4bc2710_PV_g4live/g4ok_gltf/699463ea0065185a7ffaf10d4935fc61/1/source/evt/g4live/torch/Time.ini: ptree is too deep

    Program received signal SIGABRT, Aborted.
    0x00007fffe2051207 in raise () from /lib64/libc.so.6
    Missing separate debuginfos, use: debuginfo-install boost-filesystem-1.53.0-27.el7.x86_64 boost-program-options-1.53.0-27.el7.x86_64 boost-regex-1.53.0-27.el7.x86_64 boost-system-1.53.0-27.el7.x86_64 expat-2.1.0-10.el7_3.x86_64 glfw-3.2.1-2.el7.x86_64 glibc-2.17-260.el7_6.3.x86_64 keyutils-libs-1.5.8-3.el7.x86_64 krb5-libs-1.15.1-37.el7_6.x86_64 libX11-1.6.5-2.el7.x86_64 libX11-devel-1.6.5-2.el7.x86_64 libXau-1.0.8-2.1.el7.x86_64 libXcursor-1.1.15-1.el7.x86_64 libXext-1.3.3-3.el7.x86_64 libXfixes-5.0.3-1.el7.x86_64 libXinerama-1.1.3-2.1.el7.x86_64 libXrandr-1.5.1-2.el7.x86_64 libXrender-0.9.10-1.el7.x86_64 libXxf86vm-1.1.4-1.el7.x86_64 libcom_err-1.42.9-13.el7.x86_64 libdrm-2.4.91-3.el7.x86_64 libgcc-4.8.5-36.el7_6.1.x86_64 libglvnd-1.0.1-0.8.git5baa1e5.el7.x86_64 libglvnd-glx-1.0.1-0.8.git5baa1e5.el7.x86_64 libicu-50.1.2-17.el7.x86_64 libselinux-2.5-14.1.el7.x86_64 libstdc++-4.8.5-36.el7_6.1.x86_64 libxcb-1.13-1.el7.x86_64 openssl-libs-1.0.2k-16.el7_6.1.x86_64 pcre-8.32-17.el7.x86_64 xerces-c-3.1.1-9.el7.x86_64 zlib-1.2.7-18.el7.x86_64
    (gdb) bt
    #0  0x00007fffe2051207 in raise () from /lib64/libc.so.6
    #1  0x00007fffe20528f8 in abort () from /lib64/libc.so.6
    #2  0x00007fffe29607d5 in __gnu_cxx::__verbose_terminate_handler() () from /lib64/libstdc++.so.6
    #3  0x00007fffe295e746 in ?? () from /lib64/libstdc++.so.6
    #4  0x00007fffe295e773 in std::terminate() () from /lib64/libstdc++.so.6
    #5  0x00007fffe295e993 in __cxa_throw () from /lib64/libstdc++.so.6
    #6  0x00007fffe4c27e0b in boost::throw_exception<boost::exception_detail::error_info_injector<boost::property_tree::ini_parser::ini_parser_error> > (e=...) at /usr/include/boost/throw_exception.hpp:67
    #7  0x00007fffe4c24f61 in boost::exception_detail::throw_exception_<boost::property_tree::ini_parser::ini_parser_error> (x=..., 
        current_function=0x7fffe4c6ddc0 <void boost::property_tree::ini_parser::write_ini<boost::property_tree::basic_ptree<std::string, std::string, std::less<std::string> > >(std::string const&, boost::property_tree::basic_ptree<std::string, std::string, std::less<std::string> > const&, int, std::locale const&)::__PRETTY_FUNCTION__> "void boost::property_tree::ini_parser::write_ini(const string&, const Ptree&, int, const std::locale&) [with Ptree = boost::property_tree::basic_ptree<std::basic_string<char>, std::basic_string<char> "..., file=0x7fffe4c6d240 "/usr/include/boost/property_tree/ini_parser.hpp", line=297) at /usr/include/boost/throw_exception.hpp:84
    #8  0x00007fffe4c24434 in boost::property_tree::ini_parser::write_ini<boost::property_tree::basic_ptree<std::string, std::string, std::less<std::string> > > (
        filename="/home/blyth/local/opticks/geocache/OKX4Test_lWorld0x4bc2710_PV_g4live/g4ok_gltf/699463ea0065185a7ffaf10d4935fc61/1/source/evt/g4live/torch/Time.ini", pt=..., flags=0, loc=...) at /usr/include/boost/property_tree/ini_parser.hpp:296
    #9  0x00007fffe4c22ae7 in BTree::saveTree (t=..., path_=0x168da40a8 "/home/blyth/local/opticks/geocache/OKX4Test_lWorld0x4bc2710_PV_g4live/g4ok_gltf/699463ea0065185a7ffaf10d4935fc61/1/source/evt/g4live/torch/Time.ini")
        at /home/blyth/opticks/boostrap/BTree.cc:36
    #10 0x00007fffe4bfb960 in BList<std::string, double>::save (this=0x7fffffffc730, path=0x168da40a8 "/home/blyth/local/opticks/geocache/OKX4Test_lWorld0x4bc2710_PV_g4live/g4ok_gltf/699463ea0065185a7ffaf10d4935fc61/1/source/evt/g4live/torch/Time.ini")
        at /home/blyth/opticks/boostrap/BList.cc:94
    #11 0x00007fffe4bfb7e8 in BList<std::string, double>::save (this=0x7fffffffc730, dir=0x26afda0 "/home/blyth/local/opticks/geocache/OKX4Test_lWorld0x4bc2710_PV_g4live/g4ok_gltf/699463ea0065185a7ffaf10d4935fc61/1/source/evt/g4live/torch", 
        name=0x1684a1a68 "Time.ini") at /home/blyth/opticks/boostrap/BList.cc:80
    #12 0x00007fffe4bfb5c0 in BList<std::string, double>::save (li=0x26913c0, dir=0x26afda0 "/home/blyth/local/opticks/geocache/OKX4Test_lWorld0x4bc2710_PV_g4live/g4ok_gltf/699463ea0065185a7ffaf10d4935fc61/1/source/evt/g4live/torch", 
        name=0x1684a1a68 "Time.ini") at /home/blyth/opticks/boostrap/BList.cc:33
    #13 0x00007fffe4c5b898 in BTimes::save (this=0x26913c0, dir=0x26afda0 "/home/blyth/local/opticks/geocache/OKX4Test_lWorld0x4bc2710_PV_g4live/g4ok_gltf/699463ea0065185a7ffaf10d4935fc61/1/source/evt/g4live/torch") at /home/blyth/opticks/boostrap/BTimes.cc:91
    #14 0x00007fffe4c5cf44 in BTimesTable::save (this=0x2690130, dir=0x26afda0 "/home/blyth/local/opticks/geocache/OKX4Test_lWorld0x4bc2710_PV_g4live/g4ok_gltf/699463ea0065185a7ffaf10d4935fc61/1/source/evt/g4live/torch")
        at /home/blyth/opticks/boostrap/BTimesTable.cc:144
    #15 0x00007fffe5683782 in OpticksProfile::save (this=0x268f6a0, dir=0x26afda0 "/home/blyth/local/opticks/geocache/OKX4Test_lWorld0x4bc2710_PV_g4live/g4ok_gltf/699463ea0065185a7ffaf10d4935fc61/1/source/evt/g4live/torch")
        at /home/blyth/opticks/optickscore/OpticksProfile.cc:117
    #16 0x00007fffe5683638 in OpticksProfile::save (this=0x268f6a0) at /home/blyth/opticks/optickscore/OpticksProfile.cc:104
    #17 0x00007fffe565caba in Opticks::saveProfile (this=0x2690e60) at /home/blyth/opticks/optickscore/Opticks.cc:329
    #18 0x0000000000405103 in main (argc=5, argv=0x7fffffffda88) at /home/blyth/opticks/okg4/tests/OKX4Test.cc:142
    (gdb) 


::

    [blyth@localhost issues]$ cd /home/blyth/local/opticks/geocache/OKX4Test_lWorld0x4bc2710_PV_g4live/g4ok_gltf/699463ea0065185a7ffaf10d4935fc61/1/source/evt/g4live/torch/
    [blyth@localhost torch]$ l
    total 0
    -rw-rw-r--. 1 blyth blyth 0 May 10 15:34 Time.ini
    [blyth@localhost torch]$ 


::

    (gdb) p m_tt
    $2 = (BTimesTable *) 0x2690130
    (gdb) p m_tt->dump()
    Too few arguments in function call.
    (gdb) p m_tt->dump("", NULL, NULL, NULL)
    No symbol "NULL" in current context.
    (gdb) p m_tt->dump("", 0, 0, 0)
    2019-05-10 15:56:35.768 INFO  [397182] [BTimesTable::dump@103]  filter: NONE
              0.000            Time      DeltaTime             VM        DeltaVM
              0.000           0.000      27165.178          0.000        484.044 : OpticksRun::OpticksRun_1166210908
              0.000           0.000          0.000          0.000          0.000 : Opticks::Opticks_0
              0.002           0.002          0.002        103.724        103.724 : OKX4Test.GGeo.INI_0
              0.006           0.008          0.006        103.724          0.000 : _OKX4Test.X4PhysicalVolume_0
              0.000           0.008          0.000        103.724          0.000 : _X4PhysicalVolume::convertMaterials_0
              0.004           0.012          0.004        103.988          0.264 : X4PhysicalVolume::convertMaterials_0
              0.045           0.057          0.045        104.256          0.268 : _X4PhysicalVolume::convertSolids_0
              0.533           0.590          0.533        108.140          3.884 : X4PhysicalVolume::convertSolids_0
              0.000           0.590          0.000        108.140          0.000 : _X4PhysicalVolume::convertStructure_0
             24.346          24.936         24.346       3270.148       3162.008 : X4PhysicalVolume::convertStructure_0
              0.000          24.936          0.000       3270.148          0.000 : OKX4Test.X4PhysicalVolume_0
             30.529          55.465         30.529       5525.776       2255.628 : _OKX4Test.OKMgr_0
              4.574          60.039          4.574      19524.332      13998.557 : OKX4Test.OKMgr_0
    $3 = void
    (gdb) 




