#ifndef __TIMER__HPP
#define __TIMER__HPP
#include <iostream>
#include <chrono>
#include <ctime>

class Timer
{
	typedef std::chrono::high_resolution_clock clock;
	std::chrono::time_point<clock> startTime, endTime;
	std::chrono::duration<double, std::milli> elapsed_time;
	std::string name;

public:
	double overAllTime;
	double hits;

	Timer(const std::string &name) : name(name), overAllTime(0), hits(0) { tic(); }
	inline void tic()
	{
		startTime = clock::now();
	}
	inline void toc()
	{
		endTime = clock::now();
		elapsed_time = endTime - startTime;

		overAllTime += elapsed_time.count();
		hits += 1.0f;
		//print();
	}

	inline void print()
	{
		if (hits > 0)
			std::cerr << name << ":" << elapsed_time.count() << "ms (Mean: " << overAllTime / hits << " ms )" << std::endl;
	}

	double mean()
	{
		return (overAllTime / hits) / 1000.; //in seconds
	}
};

#endif
