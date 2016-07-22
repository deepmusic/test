#ifndef PVA_DL_PROFILE_H
#define PVA_DL_PROFILE_H

#ifdef _MSC_VER
  #include <time.h>
#else
  #include <sys/time.h>
#endif

long int tic(void);
long int toc(long int timestamp_start);
void update_mean_time(double* const p_mean_time, long int elapsed_time);

#endif // end PVA_DL_PROFILE_H
