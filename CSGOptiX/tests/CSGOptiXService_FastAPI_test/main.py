"""
main.py
========

make_array_response
    convert




"""


import numpy as np
from typing import Annotated
from pydantic import BaseModel
from fastapi import FastAPI, Request, Response, Header, Depends, HTTPException

import opticks_CSGOptiX as cx


_svc = cx._CSGOptiXService()

#def do_simulation( gs ):
#    global _svc
#    print("repr(_svc)\n", repr(_svc))
#    print("_svc\n", _svc)
#    ht = _svc.simulate(gs)   ## NB this is wrapper which handles numpy arrays
#    return ht

app = FastAPI()



def make_numpy_ndarray_response( a: np.ndarray, level : int = 0, media_type : str = "application/octet-stream" ):
    """
    https://stackoverflow.com/questions/15879315/what-is-the-difference-between-ndarray-and-array-in-numpy

    numpy.array is just a convenience function to create an ndarray; it is not a class itself.
    """
    headers = {}
    headers["x-numpy-level"] = str(level)
    headers["x-numpy-dtype"] = a.dtype.name
    headers["x-numpy-shape"] = str(a.shape)
    return Response(a.tobytes('C'), headers=headers, media_type=media_type )


async def parse_request_to_numpy_ndarray(request: Request):
    """
    :param request: FastAPI Request
    :return arr: NumPy array

    Uses request body and headers with array dtype and shape to reconstruct the uploaded NumPy array
    """
    token_ = request.headers.get('x-numpy-token')
    if token_ != "secret":
        raise HTTPException(status_code=401, detail="x-numpy-token invalid")
    pass

    level_ = request.headers.get('x-numpy-level','0')
    level = int(level_)

    if level > 0:
        print("[parse_request_to_numpy_ndarray")
        print("request\n", request)
        print("request.url\n",request.url)
        print("request.headers\n",request.headers)
    pass

    dtype_ = request.headers.get('x-numpy-dtype')
    shape_ = request.headers.get('x-numpy-shape')
    type_ = request.headers.get('content-type')

    dtype = getattr(np, dtype_, None)
    shape = tuple(map(int,filter(None,map(str.strip,shape_.replace("(","").replace(")","").split(",")))))
    field = "upload"  # needs to match field name from client

    if type_.startswith("multipart/form-data"):
        form = await request.form()
        filename = form[field].filename
        contents = await form[field].read()
        a0 = np.frombuffer(contents, dtype=dtype ).reshape(*shape)
    else:
        filename = None
        data: bytes = await request.body()
        a0 = np.frombuffer(data, dtype=dtype ).reshape(*shape)
    pass
    if level == 10:
         a = a0
         ## without the copy get runtime type error in the nanobind call across the C++ python barrier
    else:
         a = a0.copy()
    pass

    if level > 0:
        # https://numpy.org/doc/stable/reference/generated/numpy.ndarray.flags.html
        print("a0.data.c_contiguous:%s" % a0.data.c_contiguous )
        print("a.data.c_contiguous:%s" % a.data.c_contiguous )
        print("a0.flags:\n", a0.flags)
        print("a.flags:\n", a.flags)
        print("content-type:%s" % type_ )
        print("filename:%s" % filename )
        print("token_[%s]" % token_ )
        print("level[%d]" % level )
        print("dtype_[%s]" % str(dtype) )
        print("shape_[%s]" % str(shape_) )
        print("shape[%s]"  % str(shape) )
        print("type(a0)\n", type(a0))
        print("type(a)\n", type(a))
        print("a[%s]" % a )
        print("]parse_request_to_numpy_ndarray")
    pass
    return a



# HMM should that have a trailing slash ?

@app.post('/array_transform', response_class=Response)
async def array_transform(a: np.ndarray = Depends(parse_request_to_numpy_ndarray)):
    """
    :param a:
    :return response: Response

    1. parse_request_as_array providing the uploaded NumPy *a*
    2. operate on *a* giving *b*
    3. return *b* as FastAPI Response

    Test this with ~/np/tests/np_curl_test/call.sh
    """

    b = a + 1

    return make_numpy_ndarray_response(b)




@app.post('/simulate', response_class=Response)
async def simulate(gs: np.ndarray = Depends(parse_request_to_numpy_ndarray)):
    """
    :param gs:
    :return response: Response

    1. parse_request_as_array providing the uploaded NumPy *gs*
    2. operate on *gs* giving *ht*
    3. return *ht* as FastAPI Response

    Test this with ~/np/tests/np_curl_test/call.sh
    """

    #ht = cx._CSGOptiXService_Simulate(gs)

    ht = _svc.simulate(gs)   ## NB this is wrapper which handles numpy arrays

    return make_numpy_ndarray_response(ht)















@app.get('/array_create', response_class=Response)
def array_create():
    """
    zeta:Downloads blyth$ curl -s -D /dev/stdout http://127.0.0.1:8000/array_create  --output arr
    HTTP/1.1 200 OK
    date: Tue, 09 Sep 2025 03:08:11 GMT
    server: uvicorn
    x-numpy-level: 0
    x-numpy-dtype: uint8
    x-numpy-shape: 512,512,3
    content-length: 786432
    content-type: application/octet-stream

    zeta:Downloads blyth$ l arr
    1536 -rw-r--r--  1 blyth  staff  786432 Sep  9 11:08 arr

    zeta:Downloads blyth$ echo $(( 512*512*3 ))
    786432
    """
    arr = array_create_()
    return make_numpy_ndarray_response( arr )




