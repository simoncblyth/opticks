# https://blog.kitware.com/cmake-target-properties/

function(echo_target_usage)
  message("Usage example for a target with name UseGLMViaBCM::")
  message("")
  message("    include(EchoTarget)")
  message("    echo_target(UseGLMViaBCM \"INTERFACE_INCLUDE_DIRECTORIES;INTERFACE_LINK_LIBRARIES;INTERFACE_FIND_PACKAGE_NAME;INTERFACE_FIND_PACKAGE_VERSION;INTERFACE_FIND_PACKAGE_EXACT\") ")   
  message("")
  message("See https://cmake.org/cmake/help/v3.4/manual/cmake-properties.7.html#target-properties for list of properties on targets")
endfunction()

function(echo_target_property tgt prop)
  get_property(v TARGET ${tgt} PROPERTY ${prop})
  get_property(d TARGET ${tgt} PROPERTY ${prop} DEFINED)
  get_property(s TARGET ${tgt} PROPERTY ${prop} SET)

  # https://cmake.org/cmake/help/v3.0/command/get_property.html
  #    If the SET option is given the variable is set to a boolean value indicating whether the property has been set
  #    If the DEFINED option is given the variable is set to a boolean value indicating whether the property has been defined such as with define_property.    
  #
  # https://cmake.org/cmake/help/v3.0/command/define_property.html
  #    ...This is primarily useful to associate documentation with property names that may be retrieved with the get_property command... 

  if(s)
    message("tgt='${tgt}' prop='${prop}' defined='${d}' set='${s}' value='${v}' ")
    message("")
  endif()
endfunction()



function(echo_target tgt props)
   if(NOT TARGET ${tgt})
    message("There is no target named '${tgt}'")
    return()
   endif()


  foreach(p ${props})
    echo_target_property("${tgt}" "${p}")
  endforeach()

  message("")

endfunction()



function(echo_target_std tgt)

    set(props
    INTERFACE_INCLUDE_DIRECTORIES
    INTERFACE_LINK_LIBRARIES
    INTERFACE_FIND_PACKAGE_NAME
    INTERFACE_FIND_PACKAGE_VERSION
    INTERFACE_FIND_PACKAGE_EXACT
    IMPORTED_LOCATION
    IMPORTED_CONFIGURATIONS
    IMPORTED_LOCATION_DEBUG
    IMPORTED_SONAME_DEBUG
    IMPORTED_LINK_DEPENDENT_LIBRARIES_DEBUG
    IMPORTED_LINK_INTERFACE_LANGUAGES_DEBUG
    IMPORTED_LINK_INTERFACE_LIBRARIES_DEBUG
    )

    message("====== tgt:${tgt} tgt_DIR:${${tgt}_DIR} ================")

    echo_target(${tgt} "${props}")
endfunction()





