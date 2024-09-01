#ifndef HAVERSINE_HEURISTIC_H
#define HAVERSINE_HEURISTIC_H

// haversine_heuristic.h
//
// Haversine (Great-Circle) Heuristic for measuring geodesic distances on Earth.
// https://en.wikipedia.org/wiki/Great-circle_distance
// Output resolution 0.1m
// @author: sahamdi
// @updated: 05/06/21
//

//

// #define Pi 0.000000017453292519943295 //#Pi/180/1000,000
// #define EARTH_RAD 6371000.7714 // mean earth radius in meter https://en.wikipedia.org/wiki/Earth_radius
// #define hscale_ 0.96 // tunes the heuristic


#include <cstdlib>
#include <math.h>

class haversine_heuristic
{
	public:
		haversine_heuristic() {};
        
        haversine_heuristic(int32_t lon1, int32_t lat1):
        lon(lon1), lat(lat1) {};

		~haversine_heuristic() {}

        inline cost_t
        get(int32_t lon1, int32_t lat1, int32_t lon2, int32_t lat2)
        {
            double a = 0.5 - cos((lat2 - lat1) * Pi)/2 + cos(lat1 * Pi) * cos(lat2 * Pi) * (1 - cos((lon2 - lon1) * Pi)) / 2;
            return floor(COEF*2*asin(sqrt(a))) ;
            // return floor(hscale_*2*EARTH_RAD*10*asin(sqrt(a))) ; //#2*R*asin... # converted to 0.1m from meter;
        }

        inline cost_t
        get(int32_t lon2, int32_t lat2)
        {
            double a = 0.5 - cos((lat2 - lat) * Pi)/2 + cos(lat * Pi) * cos(lat2 * Pi) * (1 - cos((lon2 - lon) * Pi)) / 2;
            return floor(COEF*asin(sqrt(a))) ;
            // return floor(hscale_*2*EARTH_RAD*10*asin(sqrt(a))) ; //#2*R*asin... # converted to 0.1m from meter;
        }

        // set_target(int32_t lon2, int32_t lat2)
        // {}
    
    private:
        int32_t lon, lat;

};


#endif

