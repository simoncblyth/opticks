#pragma once

#define USECUDA_API  __attribute__ ((visibility ("default")))

USECUDA_API int UseCUDA_query_device(int argc, char** argv);

