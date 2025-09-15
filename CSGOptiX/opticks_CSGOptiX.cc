/**
opticks_CSGOptiX.cc
=====================

**/

#include "NP_nanobind.h"
#include "CSGOptiXService.h"

namespace nb = nanobind;

// First argument is module name which must match the first arg to nanobind_add_module in CMakeLists.txt
NB_MODULE(opticks_CSGOptiX, m)
{
    m.doc() = "nanobind python wrapper for CSGOptiXService ";

    nb::class_<CSGOptiXService>(m,"CSGOptiXService")
        .def(nb::init<>())
        .def("__repr__", [](const CSGOptiXService& svc) { return svc.desc(); })
        .def("simulate", [](const CSGOptiXService& svc, nb::ndarray<nb::numpy>& _gs) {

            NP* gs = NP_nanobind::NP_copy_of_numpy_array(_gs);
            NP* ht = svc.simulate(gs);
            nb::ndarray<nb::numpy> _ht = NP_nanobind::numpy_array_view_of_NP(ht);

            return _ht ;
          })
        ;

}


