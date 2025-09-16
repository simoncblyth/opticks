runtime_incompatible_function_arguments_FastAPI_nanobind_issue_perhaps
=======================================================================

FIXED Issue
------------

nanobind python extension that works OK directly from python/ipython 
does not work when integrated with FastAPI. Giving runtime error

Traced this to be due to memory details of the array, either OWNDATA or WRITEABLE 
are the critical ones::

    In [8]: a.flags
    Out[8]: 
      C_CONTIGUOUS : True
      F_CONTIGUOUS : False
      OWNDATA : False
      WRITEABLE : True
      ALIGNED : True
      WRITEBACKIFCOPY : False

    In [9]: a1 = a.copy()

    In [10]: a1.flags
    Out[10]: 
      C_CONTIGUOUS : True
      F_CONTIGUOUS : False
      OWNDATA : True
      WRITEABLE : True
      ALIGNED : True
      WRITEBACKIFCOPY : False



::


    _svc
     -CSGOptiXService::desc evt YES fd YES cx YES

          INFO   127.0.0.1:60286 - "POST /simulate HTTP/1.1" 500
         ERROR   Exception in ASGI application
    Traceback (most recent call last):
      File "/home/blyth/miniconda3/envs/ok/lib/python3.13/site-packages/uvicorn/protocols/http/httptools_impl.py", line 409, in run_asgi
        result = await app(  # type: ignore[func-returns-value]
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
            self.scope, self.receive, self.send
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        )
        ^
      File "/home/blyth/miniconda3/envs/ok/lib/python3.13/site-packages/uvicorn/middleware/proxy_headers.py", line 60, in __call__
        return await self.app(scope, receive, send)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/home/blyth/miniconda3/envs/ok/lib/python3.13/site-packages/fastapi/applications.py", line 1054, in __call__
        await super().__call__(scope, receive, send)
      File "/home/blyth/miniconda3/envs/ok/lib/python3.13/site-packages/starlette/applications.py", line 113, in __call__
        await self.middleware_stack(scope, receive, send)
      File "/home/blyth/miniconda3/envs/ok/lib/python3.13/site-packages/starlette/middleware/errors.py", line 186, in __call__
        raise exc
      File "/home/blyth/miniconda3/envs/ok/lib/python3.13/site-packages/starlette/middleware/errors.py", line 164, in __call__
        await self.app(scope, receive, _send)
      File "/home/blyth/miniconda3/envs/ok/lib/python3.13/site-packages/starlette/middleware/exceptions.py", line 63, in __call__
        await wrap_app_handling_exceptions(self.app, conn)(scope, receive, send)
      File "/home/blyth/miniconda3/envs/ok/lib/python3.13/site-packages/starlette/_exception_handler.py", line 53, in wrapped_app
        raise exc
      File "/home/blyth/miniconda3/envs/ok/lib/python3.13/site-packages/starlette/_exception_handler.py", line 42, in wrapped_app
        await app(scope, receive, sender)
      File "/home/blyth/miniconda3/envs/ok/lib/python3.13/site-packages/starlette/routing.py", line 716, in __call__
        await self.middleware_stack(scope, receive, send)
      File "/home/blyth/miniconda3/envs/ok/lib/python3.13/site-packages/starlette/routing.py", line 736, in app
        await route.handle(scope, receive, send)
      File "/home/blyth/miniconda3/envs/ok/lib/python3.13/site-packages/starlette/routing.py", line 290, in handle
        await self.app(scope, receive, send)
      File "/home/blyth/miniconda3/envs/ok/lib/python3.13/site-packages/starlette/routing.py", line 78, in app
        await wrap_app_handling_exceptions(app, request)(scope, receive, send)
      File "/home/blyth/miniconda3/envs/ok/lib/python3.13/site-packages/starlette/_exception_handler.py", line 53, in wrapped_app
        raise exc
      File "/home/blyth/miniconda3/envs/ok/lib/python3.13/site-packages/starlette/_exception_handler.py", line 42, in wrapped_app
        await app(scope, receive, sender)
      File "/home/blyth/miniconda3/envs/ok/lib/python3.13/site-packages/starlette/routing.py", line 75, in app
        response = await f(request)
                   ^^^^^^^^^^^^^^^^
      File "/home/blyth/miniconda3/envs/ok/lib/python3.13/site-packages/fastapi/routing.py", line 302, in app
        raw_response = await run_endpoint_function(
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        ...<3 lines>...
        )
        ^
      File "/home/blyth/miniconda3/envs/ok/lib/python3.13/site-packages/fastapi/routing.py", line 215, in run_endpoint_function
        return await run_in_threadpool(dependant.call, **values)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/home/blyth/miniconda3/envs/ok/lib/python3.13/site-packages/starlette/concurrency.py", line 38, in run_in_threadpool
        return await anyio.to_thread.run_sync(func)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/home/blyth/miniconda3/envs/ok/lib/python3.13/site-packages/anyio/to_thread.py", line 56, in run_sync
        return await get_async_backend().run_sync_in_worker_thread(
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
            func, args, abandon_on_cancel=abandon_on_cancel, limiter=limiter
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        )
        ^
      File "/home/blyth/miniconda3/envs/ok/lib/python3.13/site-packages/anyio/_backends/_asyncio.py", line 2476, in run_sync_in_worker_thread
        return await future
               ^^^^^^^^^^^^
      File "/home/blyth/miniconda3/envs/ok/lib/python3.13/site-packages/anyio/_backends/_asyncio.py", line 967, in run
        result = context.run(func, *args)
      File "/home/blyth/opticks/CSGOptiX/tests/CSGOptiXService_FastAPI_test/main.py", line 118, in simulate
        ht = do_simulation(gs)
      File "/home/blyth/opticks/CSGOptiX/tests/CSGOptiXService_FastAPI_test/main.py", line 15, in do_simulation
        ht = _svc.simulate(gs)   ## NB this is wrapper which handles numpy arrays
    TypeError: simulate(): incompatible function arguments. The following argument types are supported:
        1. simulate(self, arg: numpy.ndarray[], /) -> numpy.ndarray[]

    Invoked with types: opticks_CSGOptiX._CSGOptiXService, ndarray



Use static to avoid exposing the struct
--------------------------------------------

::

              ^^^^^^^^^^^^
      File "/home/blyth/miniconda3/envs/ok/lib/python3.13/site-packages/anyio/_backends/_asyncio.py", line 967, in run
        result = context.run(func, *args)
      File "/home/blyth/opticks/CSGOptiX/tests/CSGOptiXService_FastAPI_test/main.py", line 120, in simulate
        ht = cx._CSGOptiXService_Simulate(gs)
    TypeError: _CSGOptiXService_Simulate(): incompatible function arguments. The following argument types are supported:
        1. _CSGOptiXService_Simulate(arg: numpy.ndarray[], /) -> numpy.ndarray[]

    Invoked with types: ndarray




Maybe need : Parameterized Wrapper Class ? 
----------------------------------------------

* https://nanobind.readthedocs.io/en/latest/api_core.html#parameterized-wrapper-classes

That seems hard to believe given that it works fine when do not use FastAPI ? 



Looking for projects using binding (pybind11/nanobind) together with FastAPI
---------------------------------------------------------------------------------

* https://github.com/tembolo1284/cpp_py_proj/blob/main/README.md


::

      1 #include <pybind11/pybind11.h>
      2 #include <pybind11/functional.h>
      3 #include "../integration_lib/include/TrapezoidalIntegration.hpp"
      4 #include "../integration_lib/include/SimpsonsIntegration.hpp"
      5 #include "../integration_lib/include/MonteCarloIntegration.hpp"
      6 #include "../integration_lib/include/AdaptiveQuadrature.hpp"
      7 #include <cmath>
      8 #include <functional>
      9 
     10 namespace py = pybind11;
     11 
     12 PYBIND11_MODULE(integration, m) {
     13     m.doc() = "Python bindings for the integration library";
     14 
     15     // For each integration method, ensure that custom functions passed from Python are thread-safe
     16 
     17     // TrapezoidalIntegration class
     18     py::class_<TrapezoidalIntegration>(m, "TrapezoidalIntegration")
     19         .def(py::init<>())  // Default constructor using x^3
     20         .def(py::init<std::function<double(double)>>())  // Constructor with custom function
     21         .def("compute", [](TrapezoidalIntegration& self, double a, double b, int nThreads) {
     22             py::gil_scoped_release release;  // Release GIL for multithreading in C++
     23             return self.compute(a, b, nThreads);
     24         }, py::call_guard<py::gil_scoped_acquire>());  // Re-acquire GIL after computation
     25 
     26     // SimpsonsIntegration class
     27     py::class_<SimpsonsIntegration>(m, "SimpsonsIntegration")
     28         .def(py::init<>())  // Default constructor using x^3
     29         .def(py::init<std::function<double(double)>>())  // Constructor with custom function
     30         .def("compute", [](SimpsonsIntegration& self, double a, double b, int nThreads) {
     31             py::gil_scoped_release release;  // Release GIL for multithreading in C++
     32             return self.compute(a, b, nThreads);
     33         }, py::call_guard<py::gil_scoped_acquire>());

