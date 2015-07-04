#pragma once

double getRealTime();
void current_time(char* buf, int buflen, const char* tfmt, int utc);
char* now(const char* tfmt, const int buflen, int utc);

