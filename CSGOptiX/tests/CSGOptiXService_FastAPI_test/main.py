"""
main.py
========

make_response
    returns fastapi.Response containing array and meta string

make_numpy_array_from_magic_bytes
    so far metadata not extracted

make_numpy_array_from_raw_bytes

parse_request
    extracting numpy array and values of some headers

simulate
    invokes C++ CSGOptiXService via _CSGOptiXService which straddles python/C++ barrier

array_create
    test endpoint

"""


import io
import numpy as np
from typing import Annotated, Optional
#from pydantic import BaseModel
from fastapi import FastAPI, Request, Response, Header, Depends, HTTPException

import opticks_CSGOptiX as cx

_svc = cx._CSGOptiXService()

app = FastAPI()


def make_response( arr:np.ndarray, meta:str="", magic:bool=False, index:int=-1, level:int = 0):
    """
    https://stackoverflow.com/questions/15879315/what-is-the-difference-between-ndarray-and-array-in-numpy

    numpy.array is just a convenience function to create an ndarray; it is not a class itself.
    """
    if level > 0:
        print("make_response arr:%s meta:%s magic:%s index:%s level:%s  " % (arr, meta,magic,index,level))
    pass
    headers = {}
    headers["x-opticks-index"] = str(index)
    headers["x-opticks-level"] = str(level)
    media_type = "application/octet-stream"

    data:bytes = b''

    if magic:
        buffer = io.BytesIO()
        np.save(buffer, arr)
        if len(meta) > 0:
            buffer.write(meta.encode("utf-8"))
        pass
        data = buffer.getvalue()
    else:
        headers["x-opticks-dtype"] = arr.dtype.name
        headers["x-opticks-shape"] = str(arr.shape)
        data = arr.tobytes('C')
    pass
    return Response(data, headers=headers, media_type=media_type )




def make_numpy_array_from_magic_bytes(data:bytes, level:int=0):
    """
    :param data: bytes which are assumed to include the numpy magic header
    :return arr: numpy.ndarray constructed from the bytes

    NB any metadata concatenated following array data is currently ignored

    TODO: follow tests/NP_nanobind_test/meta_check.py to get meta from the bytes too

    Used from parse_request when magic enabled
    """
    buffer = io.BytesIO(data)
    buffer.seek(0)
    arr = np.load(buffer)
    return arr

def make_numpy_array_from_magic_bytes_with_meta(data:bytes, level:int=0):
    """
    Test for this in ~/np/tests/NP_nanobind_test/meta_check.py
    """

    if level > 0:print(f"[make_numpy_array_from_magic_bytes_with_meta")

    buffer = io.BytesIO(data)
    buffer.seek(0)
    arr = np.load(buffer)

    buf_nbytes = len(data) # buffer.getbuffer().nbytes
    hdr_nbytes = data.find(b'\n') + 1  # 1 + index of first newline, typically 128 but can be more for arrays with many dimensions
    arr_nbytes = arr.nbytes
    meta_nbytes = buf_nbytes - hdr_nbytes - arr_nbytes

    buffer.seek( hdr_nbytes + arr_nbytes )
    _meta:Optional[bytes] = buffer.read(meta_nbytes) if meta_nbytes>0 else None
    meta = _meta.decode("utf-8")

    if level > 0:print(f"-make_numpy_array_from_magic_bytes_with_meta buf_nbytes:{buf_nbytes} hdr_nbytes:{hdr_nbytes} arr_nbytes:{arr_nbytes} meta_nbytes:{meta_nbytes} meta:{meta} ")
    if level > 0:print(f"]make_numpy_array_from_magic_bytes_with_meta")

    return arr, meta





def make_numpy_array_from_raw_bytes(data:bytes, dtype_:str, shape_:str, level:int=0 ):
    """
    :param data: bytes which are assumed to just carry array data, no header or metadata
    :param dtype_: str
    :param shape_: str

    Used from parse_request when magic not enabled

    raw bytes require dtype and shape metadata strings
    """
    dtype = getattr(np, dtype_, None)
    shape = tuple(map(int,filter(None,map(str.strip,shape_.replace("(","").replace(")","").split(",")))))
    a0 = np.frombuffer(data, dtype=dtype ).reshape(*shape)
    return a0



async def parse_request(request: Request):
    """
    :param request: FastAPI Request
    :return arr: NumPy array

    Uses request body and headers with array dtype and shape to reconstruct the uploaded NumPy array

    TODO: handle metadata concatenated after the array data
    """
    token_ = request.headers.get('x-opticks-token')
    if token_ != "secret":
        raise HTTPException(status_code=401, detail="x-opticks-token invalid")
    pass

    level_ = request.headers.get('x-opticks-level','0')
    level = int(level_)
    index_ = request.headers.get('x-opticks-index','0')
    index = int(index_)
    type_ = request.headers.get('content-type')

    if level > 0:
        print("[parse_request")
        print("request\n", request)
        print("request.url\n",request.url)
        print("request.headers\n",request.headers)
    pass

    data:bytes = b''
    if type_.startswith("multipart/form-data"):
        field = "upload"  # needs to match field name from client
        form = await request.form()
        filename = form[field].filename
        data = await form[field].read()
    else:
        filename = None
        data = await request.body()
    pass

    numpy_magic = b'\x93NUMPY'
    has_numpy_magic = data.startswith(numpy_magic)
    if has_numpy_magic:
        dtype_ = None
        shape_ = None
        #arr0 = make_numpy_array_from_magic_bytes(data, level)
        arr0, meta = make_numpy_array_from_magic_bytes_with_meta(data, level)
    else:
        dtype_ = request.headers.get('x-opticks-dtype','')
        shape_ = request.headers.get('x-opticks-shape','')
        arr0 = make_numpy_array_from_raw_bytes(data, dtype_, shape_, level )
        meta = None
    pass
    arr = arr0 if level == 10 else arr0.copy()
    ## without the copy get runtime type error in the nanobind call across the C++ python barrier

    if level > 0:
        # https://numpy.org/doc/stable/reference/generated/numpy.ndarray.flags.html
        print("has_numpy_magic:%s" % has_numpy_magic )
        print("arr0.data.c_contiguous:%s" % arr0.data.c_contiguous )
        print("arr.data.c_contiguous:%s" % arr.data.c_contiguous )
        #print("arr0.flags:\n", arr0.flags)
        #print("arr.flags:\n", arr.flags)
        print("content-type:%s" % type_ )
        print("filename:%s" % filename )
        print("token_[%s]" % token_ )
        print("level[%d]" % level )
        print("index[%d]" % index )
        print("dtype_[%s]" % str(dtype_) )
        print("shape_[%s]" % str(shape_) )
        print("type(arr0)\n", type(arr0))
        print("type(arr)\n", type(arr))
        print("arr[%s]" % arr )
        print("meta[%s]" % meta )
        print("]parse_request")
    pass
    request.state.array = arr
    request.state.meta = meta
    request.state.index = index
    request.state.level = level
    request.state.magic = has_numpy_magic




@app.post('/simulate', response_class=Response, dependencies=[Depends(parse_request)])
async def simulate(request: Request):
    """
    :param request: Request
    :return response: Response

    1. parse_request dependency sets request.state values
    2. call _svc with *gs* to give *ht*
    3. return *ht* as FastAPI Response

    Test this with ~/np/tests/np_curl_test/np_curl_test.sh
    """
    gs = request.state.array
    gs_meta = request.state.meta
    index = request.state.index
    level = request.state.level
    magic = request.state.magic

    if level > 0: print("main.py:simulate index %d gs %s gs_meta[%s] " % ( index, repr(gs), gs_meta ))

    #ht = _svc.simulate(gs, index)   ## NB this wrapper from CSGOptiX/opticks_CSGOptiX handles numpy<=>NP conversion
    ht, ht_meta = _svc.simulate_with_meta(gs, gs_meta, index)

    response = make_response(ht, ht_meta, magic, index, level)

    return response




@app.get('/array_create', response_class=Response)
def array_create():
    """
    ::

       curl -s -D /dev/stdout http://127.0.0.1:8000/array_create  --output arr

    """
    arr = np.arange(32, dtype=np.float32).reshape(2,4,4)
    meta:str = ""
    magic:bool = True
    index:int = 1
    level:int = 1

    response = make_response( arr, meta, magic, index, level )
    return response




