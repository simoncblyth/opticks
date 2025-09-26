/**
opticks_CSGOptiX.cc
=====================

**/

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/ndarray.h>
#include "NP_nanobind.h"

#include "ssys.h"
#include "OPTICKS_LOG.hh"
#include "CSGOptiXService.h"

namespace nb = nanobind;


struct _CSGOptiXService
{
   int             level ;
   CSGOptiXService svc ;

   _CSGOptiXService();
   virtual ~_CSGOptiXService();

   nb::ndarray<nb::numpy> simulate( nb::ndarray<nb::numpy> _gs, int eventID ) ;
   nb::tuple    simulate_with_meta( nb::ndarray<nb::numpy> _gs, nb::str _gs_meta, int eventID ) ;

   std::string desc() const ;
};

inline _CSGOptiXService::_CSGOptiXService()
    :
    level(ssys::getenvint("_CSGOptiXService_level",0)),
    svc()
{
    OPTICKS_ELOG("_CSGOptiXService");
    if(level > 0) std::cout << "-_CSGOptiXService::_CSGOptiXService level[" << level << "]\n" ;
}

inline _CSGOptiXService::~_CSGOptiXService()
{
    if(level > 0) std::cout << "-_CSGOptiXService::~_CSGOptiXService\n" ;
}

/**
_CSGOptiXService::simulate
---------------------------

1. convert (nb::ndarray)_gs [python argument, eg obtained from FastAPI HTTP POST request] to (NP)gs for C++ usage
2. invoke CSGOptiXService::simulate yielding (NP)ht
3. convert (NP)ht to (nb::ndarray)_ht and return that to python

Q: how to transmit metadata, eg with the hits ?


**/


inline nb::ndarray<nb::numpy> _CSGOptiXService::simulate( nb::ndarray<nb::numpy> _gs, int eventID )
{
    if(level > 0) std::cout << "[_CSGOptiXService::simulate eventID " << eventID << "\n" ;
    NP* gs = NP_nanobind::NP_copy_of_numpy_array(_gs);

    NP* ht = svc.simulate(gs, eventID );

    nb::ndarray<nb::numpy> _ht = NP_nanobind::numpy_array_view_of_NP(ht);

    if(level > 0) std::cout << "]_CSGOptiXService::simulate eventID " << eventID << "\n" ;
    return _ht ;
}

inline nb::tuple _CSGOptiXService::simulate_with_meta( nb::ndarray<nb::numpy> _gs, nb::str _gs_meta, int eventID )
{
    if(level > 0) std::cout << "[_CSGOptiXService::simulate_with_meta eventID " << eventID << "\n" ;
    NP* gs = NP_nanobind::NP_copy_of_numpy_array_with_meta(_gs, _gs_meta);

    if(level > 0) std::cout << "-_CSGOptiXService::simulate_with_meta gs.meta[" << gs->meta << "]\n" ;

    NP* ht = svc.simulate(gs, eventID );

    nb::tuple _ht = NP_nanobind::numpy_array_view_of_NP_with_meta(ht);

    if(level > 0) std::cout << "]_CSGOptiXService::simulate_with_meta eventID " << eventID << "\n" ;
    return _ht ;
}


inline std::string _CSGOptiXService::desc() const
{
    return svc.desc();
}


// First argument is module name which must match the first arg to nanobind_add_module in CMakeLists.txt
NB_MODULE(opticks_CSGOptiX, m)
{
    m.doc() = "nanobind _CSGOptiXService ";

    nb::class_<_CSGOptiXService>(m, "_CSGOptiXService")
        .def(nb::init<>())
        .def("__repr__", &_CSGOptiXService::desc)
        .def("simulate", &_CSGOptiXService::simulate )
        .def("simulate_with_meta", &_CSGOptiXService::simulate_with_meta )
        ;
}


