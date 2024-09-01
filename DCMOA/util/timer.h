#ifndef TIMER_H
#define TIMER_H

// timer.h
//
// A cross-platform monotonic wallclock timer.
// Currently supports nanoseconds resolution.
//
// Reference doco for timers on OSX:
// https://developer.apple.com/library/mac/qa/qa1398/_index.html
// https://developer.apple.com/library/mac/technotes/tn2169/_index.html#//apple_ref/doc/uid/DTS40013172-CH1-TNTAG5000
//
// @author: dharabor
//
// @updated: 05/06/2021
//

// #ifndef WARTHOG_TIMER_H
// #define WARTHOG_TIMER_H

#ifdef OS_MAC
#include <mach/mach.h>
#include <mach/mach_time.h>

#else
#include <time.h>
#endif

class timer
{

#ifdef OS_MAC
	uint64_t start_time;
	uint64_t stop_time;
	mach_timebase_info_data_t timebase;
#else
	timespec stop_time;
	timespec start_time;
#endif

public:
	timer()
	{

#ifdef OS_MAC
		start_time = stop_time = 0;
		mach_timebase_info(&timebase);

#else
		start_time.tv_sec = 0;
		start_time.tv_nsec = 0;
		stop_time.tv_sec = 0;
		stop_time.tv_nsec = 0;
#endif
	}

	double
	get_time_nano()
	{
#ifdef OS_MAC
		uint64_t raw_time = mach_absolute_time();
		return (double)(raw_time * timebase.numer / timebase.denom);
#else
		timespec raw_time;
		clock_gettime(CLOCK_MONOTONIC, &raw_time);
		return (double)(raw_time.tv_sec * 1e09 + raw_time.tv_nsec);
#endif
	}

	void start()
	{
#ifdef OS_MAC
		start_time = mach_absolute_time();
		stop_time = start_time;
#else
		clock_gettime(CLOCK_MONOTONIC, &start_time);
		stop_time = start_time;
#endif
	}

	void stop()
	{
#ifdef OS_MAC
		stop_time = mach_absolute_time();
#else
		clock_gettime(CLOCK_MONOTONIC, &stop_time);
#endif
	}

	double elapsed_time_nano()
	{
#ifdef OS_MAC
		uint64_t elapsed_time = stop_time - start_time;
		return (double)(elapsed_time * timebase.numer / timebase.denom);
		//Nanoseconds nanosecs = AbsoluteToNanoseconds(*(AbsoluteTime*)&elapsed_time);
		//return (double) UnsignedWideToUInt64(nanosecs) ;

#else
		return (stop_time.tv_sec * 1e09 + stop_time.tv_nsec) -
			   (start_time.tv_sec * 1e09 + start_time.tv_nsec);
#endif
	}

	void reset()
	{
#ifdef OS_MAC
		start_time = stop_time = 0;
#else
		start_time.tv_sec = 0;
		start_time.tv_nsec = 0;
		stop_time.tv_sec = 0;
		stop_time.tv_nsec = 0;
#endif
	}

	inline double
	elapsed_time_micro() {return elapsed_time_nano() / 1000.0;}

	inline double
	elapsed_time_second() {return elapsed_time_nano() / 1e9;}

	inline double
	get_time_micro() { return get_time_nano() / 1000.0; }
};

#endif