# These GPU and compute capability detectors are not worth 
# the CMakeLists.txt real estate they occupy.
#
# Opticks users need to know if they have a GPU and 
# they need to know its COMPUTE_CAPABILITY at configuration time.
# 
#
#
# Try to detect GPU available
# * https://github.com/visore/Visore/blob/master/code/cmake/DetectGPU.cmake
# 
# Just use "nvidia-smi -L" to detect
if(NOT DEFINED GPU)
    find_program(_nvidia_smi "nvidia-smi")
    # message(${_nvidia_smi})
    if (_nvidia_smi)
        exec_program(${_nvidia_smi} ARGS -L
                    OUTPUT_VARIABLE _nvidia_smi_out
                    RETURN_VALUE _nvidia_smi_ret)
        message(${_nvidia_smi_out})
        # message(${_nvidia_smi_ret})
        if (_nvidia_smi_ret EQUAL 0)
            # convert string with newlines to list of strings
            string(REGEX REPLACE "\n" ";" _nvidia_smi_out "${_nvidia_smi_out}")
            foreach(_line ${_nvidia_smi_out})
                if (_line MATCHES "^GPU [0-9]+:")
                    #math(EXPR DETECT_GPU_COUNT_NVIDIA_SMI "${DETECT_GPU_COUNT_NVIDIA_SMI}+1")
                    # the UUID is not very useful for the user, remove it
                    string(REGEX REPLACE " \\(UUID:.*\\)" "" _gpu_info "${_line}")
                    # message(${_gpu_info})
                    if (NOT _gpu_info STREQUAL "")
                        #list(APPEND DETECT_GPU_INFO "${_gpu_info}")
                        set(GPU ON)
                        message("Find GPU.")
                    endif()
                endif()
            endforeach()
        endif()
    endif()
endif()

# Detect compute capability using cudarap- executable 
# (doesnt handle multi-GPU)
#
# from thrustrap- tests, 
# but not worth the complication of using : 
# users need to know their COMPUTE_CAPABILITY 

if(NOT DEFINED SM)
    find_program(_exe "cudaGetDevicePropertiesTest")
    if (_exe)
        exec_program(${_exe} ARGS q OUTPUT_VARIABLE _out RETURN_VALUE _rc)
        message("${name}.${_exe} : ${_out}")
        if (_rc EQUAL 0)
           if (_out MATCHES "^[0-9]+")
              set(SM ${_out})
           endif()
        endif()
    endif()
endif()

