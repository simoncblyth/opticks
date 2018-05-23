
set(OpticksXercesC_MODULE "${CMAKE_CURRENT_LIST_FILE}")

find_path(OpticksXercesC_INCLUDE_DIR 
   NAMES "xercesc/parsers/SAXParser.hpp"
   PATHS 
      /usr/include 
      /usr/local/include
      /opt/local/include
)

find_library(OpticksXercesC_LIBRARY
   NAMES xerces-c 
   PATHS
     /usr/lib 
     /usr/lib64
     /usr/local/lib
     /opt/local/lib
)

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
    )
endif()



