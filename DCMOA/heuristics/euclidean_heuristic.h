#ifndef EUCLIDEAN_HEURISTIC_H
#define EUCLIDEAN_HEURISTIC_H

// euclidean_heuristic.h
//
// Straight-line heuristic for measuring distances in the plane.
//
// @author: dharabor, sahamdi
// @updated: 05/06/21
//

#include <math.h>
class euclidean_heuristic
{
	public:
		euclidean_heuristic() {};
        
        euclidean_heuristic(int32_t x1, int32_t y1):
        x(x1), y(y1) {};

		~euclidean_heuristic() {}

        inline cost_t
        get(int32_t x1, int32_t y1, int32_t x2, int32_t y2)
        {
            double dx = x1-x2;
            double dy = y1-y2;
            return floor(sqrt(dx*dx + dy*dy) * hscale);
        }

        inline cost_t
        get(int32_t x2, int32_t y2)
        {
            double dx = x-x2;
            double dy = y-y2;
            return floor(sqrt(dx*dx + dy*dy) * hscale);
        }

        // set_target(int32_t lon2, int32_t lat2)
        // {}
    
    private:
        int32_t x, y;
        double hscale = 1;

};


#endif

