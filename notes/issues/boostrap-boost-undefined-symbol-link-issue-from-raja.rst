boostrap-boost-undefined-symbol-link-issue-from-raja
=====================================================


::

    Hi Simon,

    I just did a git pull to re-compile the software. opticks-foreign-install went through fine. After a "opticks-wipe", an "opticks-full" fails in boostrap with the following (series of) error. Any thoughts on what could be going wrong? It was all working fine a couple of weeks ago...

    Thanks and Cheers,
    Raja.

    [ 63%] Linking CXX executable BListTest
    ../libBoostRap.so: undefined reference to `boost::program_options::detail::cmdline::cmdline(std::vector<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::allocator<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > > > const&)'
    ../libBoostRap.so: undefined reference to `boost::re_detail_107000::perl_matcher<__gnu_cxx::__normal_iterator<char const*, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<boost::sub_match<__gnu_cxx::__normal_iterator<char const*, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > > > >, boost::regex_traits<char, boost::cpp_regex_traits<char> > >::find()'
    ../libBoostRap.so: undefined reference to `boost::program_options::options_description::options_description(std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&, unsigned int, unsigned int)'
    ../libBoostRap.so: undefined reference to `boost::re_detail_107000::perl_matcher<__gnu_cxx::__normal_iterator<char const*, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<boost::sub_match<__gnu_cxx::__normal_iterator<char const*, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > > > >, boost::regex_traits<char, boost::cpp_regex_traits<char> > >::match()'
    ../libBoostRap.so: undefined reference to `boost::re_detail_107000::perl_matcher<__gnu_cxx::__normal_iterator<char const*, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<boost::sub_match<__gnu_cxx::__normal_iterator<char const*, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > > > >, boost::regex_traits<char, boost::cpp_regex_traits<char> > >::construct_init(boost::basic_regex<char, boost::regex_traits<char, boost::cpp_regex_traits<char> > > const&, boost::regex_constants::_match_flags)'
    ../libBoostRap.so: undefined reference to `boost::program_options::to_internal(std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&)'
    collect2: error: ld returned 1 exit status
    make[2]: *** [tests/BTimeKeeperTest] Error 1
    make[1]: *** [tests/CMakeFiles/BTimeKeeperTest.dir/all] Error 2
    make[1]: *** Waiting for unfinished jobs....




* https://stackoverflow.com/questions/40225571/regex-search-hpp56-undefined-reference-to-boostre-detail-106100perl-match


Probable cause is a difference between the compiler version or options used to compile the 
boost libs and that used to compile the opticks/boostrap lib.  
Because this worked for you before, it is possible that somehow different boost libs 
are being picked up by the build.

To investigate this you should use "ldd" to see the libs and "nm" 
to try to match symbols between the libs. Often doing this 
makes you realise what has changed and gone wrong.

Some example such commandlines are below.

Check the libs for BTimeKeeperTest, the source of which is opticks/boostrap/tests/BTimeKeeperTest.cc::

    [blyth@localhost lib]$ ldd BTimeKeeperTest 
        linux-vdso.so.1 =>  (0x00007ffe405b7000)
        libBoostRap.so => /home/blyth/local/opticks/lib/./../lib64/libBoostRap.so (0x00007f1682207000)
        libSysRap.so => /home/blyth/local/opticks/lib/./../lib64/libSysRap.so (0x00007f1681fce000)
        libOKConf.so => /home/blyth/local/opticks/lib/./../lib64/libOKConf.so (0x00007f1681dca000)
        libboost_system.so.1.72.0 => /home/blyth/junotop/ExternalLibs/Boost/1.72.0/lib/libboost_system.so.1.72.0 (0x00007f1681bc8000)
        libboost_program_options.so.1.72.0 => /home/blyth/junotop/ExternalLibs/Boost/1.72.0/lib/libboost_program_options.so.1.72.0 (0x00007f1681954000)
        libboost_filesystem.so.1.72.0 => /home/blyth/junotop/ExternalLibs/Boost/1.72.0/lib/libboost_filesystem.so.1.72.0 (0x00007f1681739000)
        libboost_regex.so.1.72.0 => /home/blyth/junotop/ExternalLibs/Boost/1.72.0/lib/libboost_regex.so.1.72.0 (0x00007f1681471000)
        ...


Check symbols used by libBoostRap.so::

    [blyth@localhost lib64]$ nm libBoostRap.so | c++filt | grep boost::program_options::detail::cmdline::cmdline
                     U boost::program_options::detail::cmdline::cmdline(std::vector<std::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::allocator<std::basic_string<char, std::char_traits<char>, std::allocator<char> > > > const&)
    [blyth@localhost lib64]$ 
    [blyth@localhost lib64]$ pwd
    /home/blyth/local/opticks/lib64
    [blyth@localhost lib64]$ 

List some more boost symbols that libBoostRap.so uses::

    blyth@localhost lib64]$ nm --extern-only libBoostRap.so | c++filt | grep perl  
                     U boost::re_detail_107200::perl_matcher<__gnu_cxx::__normal_iterator<char const*, std::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<boost::sub_match<__gnu_cxx::__normal_iterator<char const*, std::basic_string<char, std::char_traits<char>, std::allocator<char> > > > >, boost::regex_traits<char, boost::cpp_regex_traits<char> > >::construct_init(boost::basic_regex<char, boost::regex_traits<char, boost::cpp_regex_traits<char> > > const&, boost::regex_constants::_match_flags)
                     U boost::re_detail_107200::perl_matcher<__gnu_cxx::__normal_iterator<char const*, std::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<boost::sub_match<__gnu_cxx::__normal_iterator<char const*, std::basic_string<char, std::char_traits<char>, std::allocator<char> > > > >, boost::regex_traits<char, boost::cpp_regex_traits<char> > >::find()
                     U boost::re_detail_107200::perl_matcher<__gnu_cxx::__normal_iterator<char const*, std::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<boost::sub_match<__gnu_cxx::__normal_iterator<char const*, std::basic_string<char, std::char_traits<char>, std::allocator<char> > > > >, boost::regex_traits<char, boost::cpp_regex_traits<char> > >::match()
                     U boost::re_detail_107200::perl_matcher<char const*, std::allocator<boost::sub_match<char const*> >, boost::regex_traits<char, boost::cpp_regex_traits<char> > >::construct_init(boost::basic_regex<char, boost::regex_traits<char, boost::cpp_regex_traits<char> > > const&, boost::regex_constants::_match_flags)
                     U boost::re_detail_107200::perl_matcher<char const*, std::allocator<boost::sub_match<char const*> >, boost::regex_traits<char, boost::cpp_regex_traits<char> > >::find()
                     U boost::re_detail_107200::perl_matcher<char const*, std::allocator<boost::sub_match<char const*> >, boost::regex_traits<char, boost::cpp_regex_traits<char> > >::match()
    [blyth@localhost lib64]$ 











