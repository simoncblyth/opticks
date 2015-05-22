set(Boost_PREFIX "/opt/local")

find_library( Boost_system_LIBRARY 
              NAMES boost_system-mt
              PATHS ${Boost_PREFIX}/lib )

find_library( Boost_thread_LIBRARY 
              NAMES boost_thread-mt
              PATHS ${Boost_PREFIX}/lib )

find_library( Boost_program_options_LIBRARY 
              NAMES boost_program_options-mt
              PATHS ${Boost_PREFIX}/lib )

find_library( Boost_log_LIBRARY 
              NAMES boost_log-mt
              PATHS ${Boost_PREFIX}/lib )

find_library( Boost_log_setup_LIBRARY 
              NAMES boost_log_setup-mt
              PATHS ${Boost_PREFIX}/lib )

find_library( Boost_filesystem_LIBRARY 
              NAMES boost_filesystem-mt
              PATHS ${Boost_PREFIX}/lib )

find_library( Boost_regex_LIBRARY 
              NAMES boost_regex-mt
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

