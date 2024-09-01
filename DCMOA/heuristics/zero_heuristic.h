#ifndef ZERO_HEURISTIC_H
#define ZERO_HEURISTIC_H

// zero_heuristic.h
//
// @author: dharabor
// @created: 2014-10-22
//

// #include <cstdlib>

class zero_heuristic
{
	public:
		zero_heuristic() {};
		zero_heuristic(int32_t x, int32_t y)
        {};
		~zero_heuristic() {}

		inline double
		get(unsigned int x, unsigned int y, unsigned int x2, unsigned int y2)
		{
            return 0;
		}

		inline double
		get(sn_id_t id, sn_id_t id2)
		{
            return 0;
		}

	private:
        
};

#endif

