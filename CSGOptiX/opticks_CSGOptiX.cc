/**
opticks_CSGOptiX.cc
=====================

**/

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/ndarray.h>
#include "NP_nanobind.h"

#include "OPTICKS_LOG.hh"
#include "CSGOptiXService.h"

namespace nb = nanobind;


struct _CSGOptiXService
{
   CSGOptiXService svc ;

   _CSGOptiXService();
   virtual ~_CSGOptiXService();

   nb::ndarray<nb::numpy> simulate( nb::ndarray<nb::numpy> _gs, int eventID ) ;
   std::string desc() const ;
};

inline _CSGOptiXService::_CSGOptiXService()
    :
    svc()
{
    OPTICKS_ELOG("_CSGOptiXService");
    std::cout << "-_CSGOptiXService::_CSGOptiXService\n" ;
}

inline _CSGOptiXService::~_CSGOptiXService()
{
    std::cout << "-_CSGOptiXService::~_CSGOptiXService\n" ;
}

inline nb::ndarray<nb::numpy> _CSGOptiXService::simulate( nb::ndarray<nb::numpy> _gs, int eventID )
{
    std::cout << "[_CSGOptiXService::simulate eventID " << eventID << "\n" ;
    NP* gs = NP_nanobind::NP_copy_of_numpy_array(_gs);

    NP* ht = svc.simulate(gs, eventID );

    nb::ndarray<nb::numpy> _ht = NP_nanobind::numpy_array_view_of_NP(ht);
    std::cout << "]_CSGOptiXService::simulate eventID " << eventID << "\n" ;
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
        ;
}


