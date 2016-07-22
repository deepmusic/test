#include "util/profile.h"

// --------------------------------------------------------------------------
// platform-specific implementation for getting timestamp
// --------------------------------------------------------------------------

#ifdef _MSC_VER
static
long int get_timestamp(void)
{
  double timer_sec = (double)clock() / CLOCKS_PER_SEC;
  return (long int)ROUND(timer_sec * (long int)1000000);
}
#else
static
long int get_timestamp(void)
{
  struct timeval timer;
  gettimeofday(&timer, 0);
  return timer.tv_sec * (long int)1000000 + timer.tv_usec;
}
#endif



// --------------------------------------------------------------------------
// unified functions for measuring elapsed time
// --------------------------------------------------------------------------

long int tic(void)
{
  return get_timestamp();
}

long int toc(long int timestamp_start)
{
  return get_timestamp() - timestamp_start;
}

void update_mean_time(double* const p_mean_time, long int elapsed_time)
{
  *p_mean_time = (*p_mean_time == 0) ? (double)elapsed_time
                 : *p_mean_time * 0.9 + (double)elapsed_time * 0.1;
}
