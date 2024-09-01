#ifndef SPHERICAL_HEURISTIC_H
#define SPHERICAL_HEURISTIC_H

// spherical_heuristic.h
//
// Spherical Heuristic for measuring geodesic distances on Earth.
// https://en.wikipedia.org/wiki/Geographical_distance
// Output resolution 0.1m
// @author: sahamdi
// @updated: 05/06/21
//

#define Pi 0.000000017453292519943295 //#Pi/180/1000,000
#define EARTH_RAD 6371000.7714 // mean earth radius in meter https://en.wikipedia.org/wiki/Earth_radius
#define hscale_ 0.96 // tunes the heuristic
#define COEF 61161607.40544

#include <cstdlib>

class spherical_heuristic
{
	public:
		spherical_heuristic() {};
        
        spherical_heuristic(int32_t lon1, int32_t lat1):
        lon(lon1), lat(lat1) {};

		~spherical_heuristic() {}

        inline cost_t
        get(int32_t lon1, int32_t lat1, int32_t lon2, int32_t lat2)
        {
            double Delta_lat_squared = pow(fabs(lat1 - lat2) * Pi, 2);
            double cos_mean_lat = cos((lat1 + lat2) * Pi / 2);
            double x = cos_mean_lat * fabs(lon1 - lon2) * Pi;

            // return floor(hscale_ * EARTH_RAD * 10 * sqrt(Delta_lat_squared + pow(x, 2)) ); // R*sqrt(...) # converted to m to 0.1m;
            return floor(COEF * sqrt(Delta_lat_squared + pow(x, 2))); // R*sqrt(...) # converted to m to 0.1m;
        }

        inline cost_t
        get(int32_t lon2, int32_t lat2)
        {
            double Delta_lat_squared = pow(fabs(lat - lat2) * Pi, 2);
            double cos_mean_lat = cos((lat + lat2) * Pi / 2);
            double x = cos_mean_lat * fabs(lon - lon2) * Pi;

            // return floor(hscale_ * EARTH_RAD * 10 * sqrt(Delta_lat_squared + pow(x, 2)) ); // R*sqrt(...) # converted to m to 0.1m;
            return floor(COEF * sqrt(Delta_lat_squared + pow(x, 2))); // R*sqrt(...) # converted to m to 0.1m;
        }

        // set_target(int32_t lon2, int32_t lat2)
        // {}
    
    private:
        int32_t lon, lat;

};


#endif