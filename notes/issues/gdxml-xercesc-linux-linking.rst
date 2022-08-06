gdxml-xercesc-linux-linking
==============================


Solution
----------

Make gdxml depend in G4 at CMake level only, just so have access to G4Persistency 
target from which to grab the consistent XercesC version. 


Symptoms
----------

1. config fails to find the G4persistency target 

::


    N[blyth@localhost gdxml]$ om-conf
    === om-one-or-all conf : gdxml           /data/blyth/junotop/opticks/gdxml                            /data/blyth/junotop/ExternalLibs/opticks/head/build/gdxml    
    -- The C compiler identification is GNU 8.3.0
    -- The CXX compiler identification is GNU 8.3.0
    -- Detecting C compiler ABI info
    -- Detecting C compiler ABI info - done
    -- Check for working C compiler: /cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc830/contrib/gcc/8.3.0/bin/gcc - skipped
    -- Detecting C compile features
    -- Detecting C compile features - done
    -- Detecting CXX compiler ABI info
    -- Detecting CXX compiler ABI info - done
    -- Check for working CXX compiler: /cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc830/contrib/gcc/8.3.0/bin/g++ - skipped
    -- Detecting CXX compile features
    -- Detecting CXX compile features - done
    -- Configuring GDXML
    -- Looking for pthread.h
    -- Looking for pthread.h - found
    -- Performing Test CMAKE_HAVE_LIBC_PTHREAD
    -- Performing Test CMAKE_HAVE_LIBC_PTHREAD - Failed
    -- Looking for pthread_create in pthreads
    -- Looking for pthread_create in pthreads - not found
    -- Looking for pthread_create in pthread
    -- Looking for pthread_create in pthread - found
    -- Found Threads: TRUE  

    -- FindOpticksXercesC.cmake. Did not find G4persistency target : so look for system XercesC or one provided by cmake arguments 
    -- looking for XercescC using XERCESC_INCLUDE_DIR or system paths 

    ## THIS USUALLY A PROBLEM 

    -- find_path looking for SAXParser.hpp yields OpticksXercesC_INCLUDE_DIR /usr/include
    -- Configuring GDXMLTest
    -- Configuring done
    -- Generating done


2. linking fails with missing XercesC symbols from version shear

::

    == opticks-setup-      nodir     append      LD_LIBRARY_PATH /home/blyth/local/opticks/externals/OptiX_700/lib
    === opticks-setup-      nodir     append      LD_LIBRARY_PATH /home/blyth/local/opticks/externals/OptiX_700/lib64
    === om-make-one : gdxml           /data/blyth/junotop/opticks/gdxml                            /data/blyth/junotop/ExternalLibs/opticks/head/build/gdxml    
    [ 57%] Building CXX object CMakeFiles/GDXML.dir/GDXML.cc.o
    [ 57%] Building CXX object CMakeFiles/GDXML.dir/GDXML_LOG.cc.o
    [ 57%] Building CXX object CMakeFiles/GDXML.dir/GDXMLWrite.cc.o
    [ 57%] Building CXX object CMakeFiles/GDXML.dir/GDXMLRead.cc.o
    [ 71%] Linking CXX shared library libGDXML.so
    [ 71%] Built target GDXML
    [ 85%] Building CXX object tests/CMakeFiles/GDXMLTest.dir/GDXMLTest.cc.o
    [100%] Linking CXX executable GDXMLTest
    CMakeFiles/GDXMLTest.dir/GDXMLTest.cc.o: In function `xercesc_3_2::DTDEntityDecl::~DTDEntityDecl()':
    /data/blyth/junotop/ExternalLibs/Xercesc/3.2.2/include/xercesc/validators/DTD/DTDEntityDecl.hpp:160: undefined reference to `xercesc_3_2::XMLEntityDecl::~XMLEntityDecl()'
    CMakeFiles/GDXMLTest.dir/GDXMLTest.cc.o: In function `xercesc_3_2::DTDEntityDecl::~DTDEntityDecl()':
    /data/blyth/junotop/ExternalLibs/Xercesc/3.2.2/include/xercesc/validators/DTD/DTDEntityDecl.hpp:162: undefined reference to `xercesc_3_2::XMemory::operator delete(void*)'
    CMakeFiles/GDXMLTest.dir/GDXMLTest.cc.o: In function `xercesc_3_2::HandlerBase::fatalError(xercesc_3_2::SAXParseException const&)':
    /data/blyth/junotop/ExternalLibs/Xercesc/3.2.2/include/xercesc/sax/HandlerBase.hpp:398: undefined reference to `xercesc_3_2::SAXParseException::SAXParseException(xercesc_3_2::SAXParseException const&)'
    /data/blyth/junotop/ExternalLibs/Xercesc/3.2.2/include/xercesc/sax/HandlerBase.hpp:398: undefined reference to `xercesc_3_2::SAXParseException::~SAXParseException()'
    /data/blyth/junotop/ExternalLibs/Xercesc/3.2.2/include/xercesc/sax/HandlerBase.hpp:398: undefined reference to `typeinfo for xercesc_3_2::SAXParseException'
    CMakeFiles/GDXMLTest.dir/GDXMLTest.cc.o:(.rodata._ZTVN11xercesc_3_213DTDEntityDeclE[_ZTVN11xercesc_3_213DTDEntityDeclE]+0x20): undefined reference to `xercesc_3_2::DTDEntityDecl::isSerializable() const'
    CMakeFiles/GDXMLTest.dir/GDXMLTest.cc.o:(.rodata._ZTVN11xercesc_3_213DTDEntityDeclE[_ZTVN11xercesc_3_213DTDEntityDeclE]+0x28): undefined reference to `xercesc_3_2::DTDEntityDecl::serialize(xercesc_3_2::XSerializeEngine&)'
    CMakeFiles/GDXMLTest.dir/GDXMLTest.cc.o:(.rodata._ZTVN11xercesc_3_213DTDEntityDeclE[_ZTVN11xercesc_3_213DTDEntityDeclE]+0x30): undefined reference to `xercesc_3_2::DTDEntityDecl::getProtoType() const'
    CMakeFiles/GDXMLTest.dir/GDXMLTest.cc.o:(.rodata._ZTVN11xercesc_3_213XMLAttDefListE[_ZTVN11xercesc_3_213XMLAttDefListE]+0x20): undefined reference to `xercesc_3_2::XMLAttDefList::isSerializable() const'
    CMakeFiles/GDXMLTest.dir/GDXMLTest.cc.o:(.rodata._ZTVN11xercesc_3_213XMLAttDefListE[_ZTVN11xercesc_3_213XMLAttDefListE]+0x28): undefined reference to `xercesc_3_2::XMLAttDefList::serialize(xercesc_3_2::XSerializeEngine&)'
    CMakeFiles/GDXMLTest.dir/GDXMLTest.cc.o:(.rodata._ZTVN11xercesc_3_213XMLAttDefListE[_ZTVN11xercesc_3_213XMLAttDefListE]+0x30): undefined reference to `xercesc_3_2::XMLAttDefList::getProtoType() const'
    CMakeFiles/GDXMLTest.dir/GDXMLTest.cc.o:(.rodata._ZTIN11xercesc_3_213DTDEntityDeclE[_ZTIN11xercesc_3_213DTDEntityDeclE]+0x10): undefined reference to `typeinfo for xercesc_3_2::XMLEntityDecl'
    ../libGDXML.so: undefined reference to `xercesc_3_2::XMLUni::fgXercescDefaultLocale'
    ../libGDXML.so: undefined reference to `xercesc_3_2::SAXParseException::getLineNumber() const'
    ../libGDXML.so: undefined reference to `xercesc_3_2::AbstractDOMParser::setDoSchema(bool)'
    ../libGDXML.so: undefined reference to `typeinfo for xercesc_3_2::XMLException'
    ../libGDXML.so: undefined reference to `xercesc_3_2::XercesDOMParser::setErrorHandler(xercesc_3_2::ErrorHandler*)'
    ../libGDXML.so: undefined reference to `xercesc_3_2::AbstractDOMParser::setDoNamespaces(bool)'
    ../libGDXML.so: undefined reference to `xercesc_3_2::AbstractDOMParser::setValidationScheme(xercesc_3_2::AbstractDOMParser::ValSchemes)'
    ../libGDXML.so: undefined reference to `xercesc_3_2::LocalFileFormatTarget::LocalFileFormatTarget(char const*, xercesc_3_2::MemoryManager*)'
    ../libGDXML.so: undefined reference to `xercesc_3_2::XMLString::release(char**, xercesc_3_2::MemoryManager*)'
    ../libGDXML.so: undefined reference to `xercesc_3_2::AbstractDOMParser::getDocument()'
    ../libGDXML.so: undefined reference to `xercesc_3_2::AbstractDOMParser::parse(char const*)'
    ../libGDXML.so: undefined reference to `xercesc_3_2::XMemory::operator new(unsigned long)'
    ../libGDXML.so: undefined reference to `xercesc_3_2::XMLPlatformUtils::fgMemoryManager'
    ../libGDXML.so: undefined reference to `typeinfo for xercesc_3_2::DOMException'
    ../libGDXML.so: undefined reference to `xercesc_3_2::XMLPlatformUtils::Initialize(char const*, char const*, xercesc_3_2::PanicHandler*, xercesc_3_2::MemoryManager*)'
    ../libGDXML.so: undefined reference to `xercesc_3_2::XMLString::transcode(char const*, char16_t*, unsigned long, xercesc_3_2::MemoryManager*)'
    ../libGDXML.so: undefined reference to `xercesc_3_2::AbstractDOMParser::setValidationSchemaFullChecking(bool)'
    ../libGDXML.so: undefined reference to `xercesc_3_2::XercesDOMParser::XercesDOMParser(xercesc_3_2::XMLValidator*, xercesc_3_2::MemoryManager*, xercesc_3_2::XMLGrammarPool*)'
    ../libGDXML.so: undefined reference to `xercesc_3_2::XMLString::transcode(char16_t const*, xercesc_3_2::MemoryManager*)'
    ../libGDXML.so: undefined reference to `xercesc_3_2::XMLUni::fgDOMWRTFormatPrettyPrint'
    ../libGDXML.so: undefined reference to `xercesc_3_2::DOMImplementationRegistry::getDOMImplementation(char16_t const*)'
    collect2: error: ld returned 1 exit status
    make[2]: *** [tests/GDXMLTest] Error 1
    make[1]: *** [tests/CMakeFiles/GDXMLTest.dir/all] Error 2
    make: *** [all] Error 2
    === om-one-or-all make : non-zero rc 2
    N[blyth@localhost gdxml]$ 

