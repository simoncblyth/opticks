/**
opticks_CSGOptiX.cc
=====================

**/

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/ndarray.h>
#include "NP_nanobind.h"

#include "CSGOptiXService.h"

namespace nb = nanobind;


struct _CSGOptiXService
{
   CSGOptiXService svc ;

   _CSGOptiXService();
   virtual ~_CSGOptiXService();

   nb::ndarray<nb::numpy> simulate( nb::ndarray<nb::numpy> _gs ) ;
   std::string desc() const ;
};

inline _CSGOptiXService::_CSGOptiXService()
    :
    svc()
{
    std::cout << "-_CSGOptiXService::_CSGOptiXService\n" ;
}

inline _CSGOptiXService::~_CSGOptiXService()
{
    std::cout << "-_CSGOptiXService::~_CSGOptiXService\n" ;
}

inline nb::ndarray<nb::numpy> _CSGOptiXService::simulate( nb::ndarray<nb::numpy> _gs )
{
    std::cout << "[_CSGOptiXService::simulate\n" ;
    NP* gs = NP_nanobind::NP_copy_of_numpy_array(_gs);

    NP* ht = svc.simulate(gs);

    nb::ndarray<nb::numpy> _ht = NP_nanobind::numpy_array_view_of_NP(ht);
    std::cout << "]_CSGOptiXService::simulate\n" ;
    return _ht ;
}

inline std::string _CSGOptiXService::desc() const
{
    return svc.desc();
}


/**
Using static CSGOptiXService::Simulate avoids the need
to expose the CSGOptiXService C++ class to python.

There was a runtime type problem with the input arra
when using FastAPI, when not copying the array derived from the request data.

**/

nb::ndarray<nb::numpy> _CSGOptiXService_Simulate( nb::ndarray<nb::numpy> _gs )
{
    NP* gs = NP_nanobind::NP_copy_of_numpy_array(_gs);

    NP* ht = CSGOptiXService::Simulate(gs);

    nb::ndarray<nb::numpy> _ht = NP_nanobind::numpy_array_view_of_NP(ht);

    return _ht ;
}


// First argument is module name which must match the first arg to nanobind_add_module in CMakeLists.txt
NB_MODULE(opticks_CSGOptiX, m)
{
    m.doc() = "nanobind _CSGOptiXService ";
    m.def("_CSGOptiXService_Simulate", &_CSGOptiXService_Simulate, nb::arg("input").sig("numpy.ndarray"), nb::sig("def _CSGOptiXService_Simulate(input: numpy.ndarray) -> numpy.ndarray"));

    nb::class_<_CSGOptiXService>(m, "_CSGOptiXService")
        .def(nb::init<>())
        .def("__repr__", &_CSGOptiXService::desc)
        .def("simulate", &_CSGOptiXService::simulate )
        ;
}












