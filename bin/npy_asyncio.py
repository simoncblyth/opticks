#!/usr/bin/env python
"""
npy_asyncio.py
----------------

TODO: integrate with external runloop/REPL, 
eg receiving an array whilst in a live ipython session

* https://ipython.readthedocs.io/en/stable/interactive/autoawait.html

https://docs.python.org/3/library/asyncio-stream.html

Async marks a function that may be interrupted, await is required to call
async-functions (aka coroutine) and marks a point were task can be switched.

* https://blog.jupyter.org/ipython-7-0-async-repl-a35ce050f7f7

::

    loop = asyncio.get_event_loop()
    loop.run_until_complete(child(10))


https://docs.python.org/3/library/asyncio-dev.html#asyncio-multithreading
https://docs.python.org/3/library/asyncio-dev.html#concurrency-and-multithreading
https://docs.python.org/3/library/asyncio-eventloop.html#asyncio.loop.run_in_executor


Similar to first example from:

* https://www.oreilly.com/library/view/using-asyncio-in/9781492075325/ch04.html


"""
import os, sys, logging, asyncio, numpy as np
from opticks.bin.npy import serialize_with_header, HEADER_BYTES, unpack_prefix
from opticks.bin.npy import npy_deserialize, meta_deserialize

log = logging.getLogger(__name__)

gethost = lambda:os.environ.get("TCP_HOST", "127.0.0.1" ) 
getport = lambda:int(os.environ.get("TCP_PORT", "15006" )) 
getdump = lambda:os.environ.get("DUMP","0")

async def npy_write( writer, arr, meta ):
    log.info(f'npy_write:{arr.shape!r}')
    if getdump() == "1":
        print(arr.shape)
        print(arr)
        print(meta)
    pass
    writer.write(serialize_with_header(arr, meta))
    await writer.drain()

async def npy_read( reader):

    prefix = await reader.readexactly(HEADER_BYTES)
    sizes = unpack_prefix(prefix)
    log.info(f"npy_read received prefix {sizes!r}")

    hdr_bytes, arr_bytes, meta_bytes, zero = sizes
    assert zero == 0 

    arr_data = await reader.readexactly(hdr_bytes+arr_bytes) 
    arr = npy_deserialize(arr_data)

    meta_data = await reader.readexactly(meta_bytes) 
    meta = meta_deserialize(meta_data)
    if getdump() == "1":
        print(arr.shape)
        print(arr)
        print(meta)
    pass
    return arr, meta

async def npy_client(arr, meta):
    reader, writer = await asyncio.open_connection(gethost(),getport())
    await npy_write( writer, arr, meta )
    arr, meta = await npy_read(reader) ;  
    log.info('npy_client : close connection')
    writer.close()
    await writer.wait_closed()


async def handle_npy(reader, writer):
    addr = writer.get_extra_info('peername')
    log.info(f"handle_npy : received from peer {addr!r}")
    arr, meta = await npy_read(reader) ;  

    meta["src"] = sys.argv[0]
    arr += 42 
    await npy_write( writer, arr, meta )

    log.info("handle_npy : close connection")
    writer.close()

async def npy_server():
    server = await asyncio.start_server(handle_npy, gethost(), getport())
    addr = server.sockets[0].getsockname()
    log.info(f'npy_server : serving on {addr}')
    async with server:
        await server.serve_forever()
    pass


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) > 1 and sys.argv[1] == "s":
        asyncio.run(npy_server())
    else:
        arr = np.zeros((1,6,4), dtype=np.float32)
        meta = dict(src=sys.argv[0])
        asyncio.run(npy_client(arr, meta))
    pass

