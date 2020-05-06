#[=[
# https://stackoverflow.com/questions/32756195/recursive-list-of-link-libraries-in-cmake

Recursive traversal over targets would allow development of an opticks-config 
that goes direct to CMake without using pkg-config and loadsa pc files.
However not sure that is so desirable.

See examples/UseG4OK for usage of this.

#]=]


function(dump_target TARGET)
    get_target_property(TYPE ${TARGET} TYPE)
    get_target_property(IMPORTED ${TARGET} IMPORTED)

    set(IMPORTED_LOCATION)
    set(LOCATION)
    if(TYPE STREQUAL "INTERFACE_LIBRARY")
    else()
       get_target_property(IMPORTED_LOCATION ${TARGET} IMPORTED_LOCATION)
       get_target_property(LOCATION          ${TARGET} LOCATION)
    endif()

    if (IMPORTED)
        get_target_property(LIBS ${TARGET} INTERFACE_LINK_LIBRARIES)
        get_target_property(INCS ${TARGET} INTERFACE_INCLUDE_DIRECTORIES)
        get_target_property(DEFS ${TARGET} INTERFACE_COMPILE_DEFINITIONS)
    else()
        get_target_property(LIBS ${TARGET} LINK_LIBRARIES)
        get_target_property(INCS ${TARGET} INCLUDE_DIRECTORIES)
        get_target_property(DEFS ${TARGET} COMPILE_DEFINITIONS)
    endif()

    if(LL_VERBOSE)
    message(STATUS "dump_target.IMPORTED:${IMPORTED} TYPE:${TYPE} TARGET:${TARGET}  ")
    message(STATUS "IMPORTED_LOCATION:${IMPORTED_LOCATION} ") 
    message(STATUS "LOCATION:${LOCATION} ") 
    message(STATUS "LIBS:${LIBS} ") 
    message(STATUS "INCS:${INCS} ") 
    message(STATUS "DEFS:${DEFS} ") 
    endif()
endfunction()


function(get_link_libraries_recursive OUTPUT_LIST TARGET)

    list(APPEND VISITED_TARGETS ${TARGET})

    dump_target(${TARGET})
    get_target_property(LIBS ${TARGET} INTERFACE_LINK_LIBRARIES)

    set(LIB_FILES "")
    foreach(LIB ${LIBS})
        if (TARGET ${LIB})
            list(FIND VISITED_TARGETS ${LIB} VISITED)
            if (${VISITED} EQUAL -1)
                
                get_target_property(LIB_TYPE ${LIB} TYPE)

                if(LL_VERBOSE)
                message(STATUS "get_link_libraries_recursive.LIB:${LIB} LIB_TYPE:${LIB_TYPE} ") 
                endif()

                set(LIB_FILE)
                if(LIB_TYPE STREQUAL "INTERFACE_LIBRARY")
                else()
                   get_target_property(LIB_FILE ${LIB} LOCATION)
                   if(LL_VERBOSE)
                   message(STATUS "get_link_libraries_recursive.LIB.LOCATION ${LIB} -> ${LIB_FILE} ")
                   endif()
                endif()

                dump_target(${LIB})

                get_link_libraries_recursive(LINK_LIB_FILES ${LIB})
                list(APPEND LIB_FILES ${LIB_FILE} ${LINK_LIB_FILES})
            endif()
        else()
            message(STATUS "non-target ${LIB}")
        endif()
    endforeach()
    set(VISITED_TARGETS ${VISITED_TARGETS} PARENT_SCOPE)
    set(${OUTPUT_LIST} ${LIB_FILES} PARENT_SCOPE)
endfunction()



#[=[
traverse
---------

Recursive traverse a tree of targets using INTERFACE_LINK_LIBRARIES

#]=]

function(traverse TARGET)
    dump_target(${TARGET})

    list(APPEND TRAVERSE_VISITED_TARGETS ${TARGET})
    get_target_property(SUBS ${TARGET} INTERFACE_LINK_LIBRARIES)

    if(SUBS)
        message(STATUS "traverse.TARGET : ${TARGET}     SUBS : ${SUBS}")
        foreach(SUB ${SUBS})
            if (TARGET ${SUB})
                list(FIND TRAVERSE_VISITED_TARGETS ${SUB} VISITED)
                if (${VISITED} EQUAL -1)
                     traverse( ${SUB} )
                endif()
            else()
                message(STATUS "traverse.non-target:${SUB}")
            endif()
        endforeach()
    else()
        message(STATUS "traverse.TARGET : ${TARGET}    no-SUBS")
    endif()
    set(TRAVERSE_VISITED_TARGETS ${TRAVERSE_VISITED_TARGETS} PARENT_SCOPE)
endfunction()



#[=[
traverse_out
--------------

Recursive traverse a tree of targets using INTERFACE_LINK_LIBRARIES, 
collecting some property PROP from each node of the tree into OUTPUT_LIST.

Note the last line, as the function completes the VALUE_LIST 
gets promoted into the parent scope OUTPUT_LIST variable, ie SUB_VALUE_LIST
during the traverse which then gets appended to the higher level VALUE_LIST
and so on as the recursion unwinds.  

For PROP LOCATION getting::

    -- traverse_out.non-target:stdc++
    -- traverse_out.non-target:/usr/local/cuda/lib/libcudart_static.a
    -- traverse_out.non-target:-Wl,-rpath,/usr/local/cuda/lib
    -- traverse_out.non-target:/usr/local/cuda/lib/libcudart_static.a
    -- traverse_out.non-target:-Wl,-rpath,/usr/local/cuda/lib

#]=]

function(traverse_out OUTPUT_LIST PROP TARGET)
    dump_target(${TARGET})

    list(APPEND TRAVERSE_OUT_VISITED_TARGETS ${TARGET})
    get_target_property(TYPE ${TARGET} TYPE)

    set(VALUE_LIST)
    set(VALUE)
    if(NOT TYPE STREQUAL "INTERFACE_LIBRARY")
       get_target_property(VALUE ${TARGET} ${PROP})
    endif()

    if(VALUE)
    list(APPEND VALUE_LIST ${VALUE})
    endif()

    get_target_property(SUBS ${TARGET} INTERFACE_LINK_LIBRARIES)

    if(SUBS)
        if(LL_VERBOSE)
        message(STATUS "traverse_out.TARGET : ${TARGET}     SUBS : ${SUBS}")
        endif()
        foreach(SUB ${SUBS})
            if (TARGET ${SUB})
                list(FIND TRAVERSE_OUT_VISITED_TARGETS ${SUB} VISITED)
                if (${VISITED} EQUAL -1)
                     traverse_out( SUB_VALUE_LIST ${PROP} ${SUB} )
                     list(APPEND VALUE_LIST ${SUB_VALUE_LIST})
                endif()
            else()
                #if(LL_VERBOSE)
                message(STATUS "traverse_out.non-target:${SUB}")
                #endif()
            endif()
        endforeach()
    else()
        if(LL_VERBOSE)
        message(STATUS "traverse_out.TARGET : ${TARGET}    no-SUBS")
        endif()
    endif()

    set(TRAVERSE_OUT_VISITED_TARGETS ${TRAVERSE_OUT_VISITED_TARGETS} PARENT_SCOPE)
    set(${OUTPUT_LIST} ${VALUE_LIST} PARENT_SCOPE)
endfunction()




#[=[

set(LL_VERBOSE ON) 
set(LINK_LIBS)
get_link_libraries_recursive( LINK_LIBS Opticks::G4OK)

foreach(LIB ${LINK_LIBS})
  message(STATUS "LIB: ${LIB} " )
endforeach()


#]=]

#[=[
traverse(Opticks::G4OK)
#]=]


#[=[
set(OUT_LIST)
traverse_out(OUT_LIST Opticks::G4OK)
foreach(OUT ${OUT_LIST})
   message(STATUS ".... traverse_out.OUT: ${OUT} " )
endforeach()
#]=]


