set(Boost_PREFIX "$ENV{BOOST_INSTALL_DIR}")
set(Boost_SUFFIX "$ENV{BOOST_SUFFIX}")

LINK_DIRECTORIES(${Boost_PREFIX}/lib)

find_library( Boost_system_LIBRARY 
              NAMES boost_system${Boost_SUFFIX}
              PATHS ${Boost_PREFIX}/lib )

find_library( Boost_thread_LIBRARY 
              NAMES boost_thread${Boost_SUFFIX}
              PATHS ${Boost_PREFIX}/lib )

find_library( Boost_program_options_LIBRARY 
              NAMES boost_program_options${Boost_SUFFIX}
              PATHS ${Boost_PREFIX}/lib )

find_library( Boost_log_LIBRARY 
              NAMES boost_log${Boost_SUFFIX}
              PATHS ${Boost_PREFIX}/lib )

find_library( Boost_log_setup_LIBRARY 
              NAMES boost_log_setup${Boost_SUFFIX}
              PATHS ${Boost_PREFIX}/lib )

find_library( Boost_filesystem_LIBRARY 
              NAMES boost_filesystem${Boost_SUFFIX}
              PATHS ${Boost_PREFIX}/lib )

find_library( Boost_regex_LIBRARY 
              NAMES boost_regex${Boost_SUFFIX}
              PATHS ${Boost_PREFIX}/lib )


set(Boost_LIBRARIES 
     ${Boost_system_LIBRARY}
     ${Boost_thread_LIBRARY}
     ${Boost_program_options_LIBRARY}
     ${Boost_log_LIBRARY}
     ${Boost_log_setup_LIBRARY}
     ${Boost_filesystem_LIBRARY}
     ${Boost_regex_LIBRARY}
)
set(Boost_INCLUDE_DIRS "${Boost_PREFIX}/include")
set(Boost_DEFINITIONS "-DBOOST_LOG_DYN_LINK" )

