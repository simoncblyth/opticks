#[=[
XercesC library and include_dir are looked for using 
three approaches

1. using Geant4 G4persistency target INTERFACE_LINK_LIBRARIES which 
   provides the absolute path to the library, the include_dir is then
   guessed to be ../include relative to this : these are the default 
   values which the below direct setting can override 

2. direct setting from CMake commandline with::

    -DXERCESC_INCLUDE_DIR=...
    -DXERCESC_LIBRARY=/path/to/the/lib.so 

3. if no direct setting is used system directories are searched, which 
   will trump those obtained from the G4persistency target.  
    

Hmm : maybe system should be the default that gets trumped by direct setting 
or G4persistency target ?

#]=]


set(OpticksXercesC_MODULE "${CMAKE_CURRENT_LIST_FILE}")

if(OpticksXercesC_VERBOSE)
message(STATUS "OpticksXercesC_MODULE : ${OpticksXercesC_MODULE} " )
endif()


#[=[
Fishing for XercesC within the G4persistency target allows to avoid
problems of picking up different versions of XercesC. 

::

   -- G4persistency.ILL : G4geometry;G4global;G4graphics_reps;G4intercoms;G4materials;G4particles;G4digits_hits;G4event;G4processes;G4run;G4track;G4tracking;/usr/lib64/libxerces-c-3.1.so

#]=]

set(xercesc_lib)
set(xercesc_include_dir)

if(TARGET Geant4::G4persistency AND TARGET XercesC::XercesC)
   # this works with Geant4 1062
   get_target_property(_lll Geant4::G4persistency INTERFACE_LINK_LIBRARIES)
   message(STATUS "FindOpticksXercesC.cmake. Found Geant4::G4persistency AND XercesC::XercesC target _lll ${_lll} " )

   get_target_property(xercesc_lib         XercesC::XercesC IMPORTED_LOCATION )
   get_target_property(xercesc_include_dir XercesC::XercesC INTERFACE_INCLUDE_DIRECTORIES )

   if(OpticksXercesC_VERBOSE)
       message(STATUS "FindOpticksXercesC.cmake. XercesC::XercesC target xercesc_lib         : ${xercesc_lib} " )
       message(STATUS "FindOpticksXercesC.cmake. XercesC::XercesC target xercesc_include_dir : ${xercesc_include_dir} " )
   endif()


elseif(TARGET G4persistency)
   # this works with Geant4 1042
    get_target_property(_lll G4persistency INTERFACE_LINK_LIBRARIES)
    message(STATUS "FindOpticksXercesC.cmake. Found G4persistency target _lll ${_lll}" )
    foreach(_lib ${_lll})
        get_filename_component(_nam ${_lib} NAME) 
        string(FIND "${_nam}" "libxerces-c" _pos ) 
        if(_pos EQUAL 0)
            #message(STATUS "_lib ${_lib}  _nam ${_nam} _pos ${_pos} ") 
            set(xercesc_lib ${_lib})
        endif()
    endforeach()

    if(xercesc_lib)
        get_filename_component(_dir ${xercesc_lib} DIRECTORY) 
        get_filename_component(_dirdir ${_dir} DIRECTORY) 
        set(xercesc_include_dir "${_dirdir}/include" )    
    endif()

    if(OpticksXercesC_VERBOSE)
       message(STATUS " G4persistency.xercesc_lib         : ${xercesc_lib} ")
       message(STATUS " G4persistency.xercesc_include_dir : ${xercesc_include_dir} ")
    endif()

else()
    #message(FATAL_ERROR "G4persistency target is required" )
    message(STATUS "FindOpticksXercesC.cmake. Did not find G4persistency target : so look for system XercesC or one provided by cmake arguments " )
endif()


set(OpticksXercesC_LIBRARY     ${xercesc_lib})
set(OpticksXercesC_INCLUDE_DIR ${xercesc_include_dir})

if(OpticksXercesC_INCLUDE_DIR AND OpticksXercesC_LIBRARY)
   set(OpticksXercesC_FOUND "YES")
else()
   set(OpticksXercesC_FOUND "NO")
endif()



#[=[
without NO_DEFAULT_PATH  this will look in CMAKE_PREFIX_PATH lib dirs 
#]=]

if(NOT OpticksXercesC_FOUND)
    message(STATUS "looking for XercescC using XERCESC_INCLUDE_DIR or system paths ")

    if(XERCESC_INCLUDE_DIR)
        set(OpticksXercesC_INCLUDE_DIR ${XERCESC_INCLUDE_DIR}) 
    else()
        find_path(OpticksXercesC_INCLUDE_DIR 
           NAMES "xercesc/parsers/SAXParser.hpp"
           PATHS 
              /usr/include 
              /usr/local/include
              /opt/local/include
           NO_DEFAULT_PATH  
        )
        message(STATUS "find_path looking for SAXParser.hpp yields OpticksXercesC_INCLUDE_DIR ${OpticksXercesC_INCLUDE_DIR}" )
    endif()


    if(XERCESC_LIBRARY) 
        set(OpticksXercesC_LIBRARY ${XERCESC_LIBRARY})
    else()
        find_library(OpticksXercesC_LIBRARY
           NAMES xerces-c 
           PATHS
             /usr/lib 
             /usr/lib64
             /usr/local/lib
             /opt/local/lib
             /usr/lib/x86_64-linux-gnu
           NO_DEFAULT_PATH  
        )
    endif()

endif()




if(OpticksXercesC_INCLUDE_DIR AND OpticksXercesC_LIBRARY)
   set(OpticksXercesC_FOUND "YES")
else()
   set(OpticksXercesC_FOUND "NO")
endif()

if(OpticksXercesC_VERBOSE)
   message(STATUS "FindOpticksXercesC.cmake OpticksXercesC_MODULE      : ${OpticksXercesC_MODULE}  ")
   message(STATUS "FindOpticksXercesC.cmake OpticksXercesC_INCLUDE_DIR : ${OpticksXercesC_INCLUDE_DIR}  ")
   message(STATUS "FindOpticksXercesC.cmake OpticksXercesC_LIBRARY     : ${OpticksXercesC_LIBRARY}  ")
   message(STATUS "FindOpticksXercesC.cmake OpticksXercesC_FOUND       : ${OpticksXercesC_FOUND}  ")
endif()

set(tgt Opticks::OpticksXercesC)
if(OpticksXercesC_FOUND AND NOT TARGET ${tgt})
    add_library(${tgt} UNKNOWN IMPORTED) 
    set_target_properties(${tgt} 
         PROPERTIES 
            IMPORTED_LOCATION             "${OpticksXercesC_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${OpticksXercesC_INCLUDE_DIR}"
            INTERFACE_PKG_CONFIG_NAME     "OpticksXercesC"
    )
endif()



