#ifndef PQUEUE_H
#define PQUEUE_H

// pqueue.h
//
// A min priority queue. Loosely based on an implementation from HOG
// by Nathan Sturtevant.
//
// @author: dharabor
// @created: 05/06/2020
//


class pqueue 
{
    
	public:
    
        pqueue(unsigned int size = 1024, cost_t** cost_array = 0, sn_id_t num_nodes = 0)
            : initsize_(size), maxsize_(size), queuesize_(0), elts_(0), cost_index_(0)
        {
            resize(size);
            priority_ = new unsigned int [num_nodes]();
            cost_ = cost_array;
        }

        ~pqueue()
        {
            delete [] elts_;
            delete [] priority_;
        }

        // removes all elements from the pqueue
        void
        clear()
        {
            queuesize_ = 0;
            // maxsize_ = initsize_;
        }

		// reprioritise the specified element (up or down)
        void 
        decrease_key(sn_id_t vertex, cost_t  f_org = 0)
        {	
            heapify_up(priority_[vertex]);
        }

        void 
        increase_key(sn_id_t vertex)
        {
            heapify_down(priority_[vertex]);
        }

		// add a new element to the pqueue
        void 
        push(sn_id_t vertex)
        {
            // if(contains(vertex))
            // {
            //     return;
            // }
            
            if(queuesize_+1 > maxsize_)
            {
                resize(maxsize_*2);
            }
            unsigned int priority = queuesize_;
            elts_[priority] = vertex;
            priority_[vertex] = priority;
            
            // val->set_priority(index_,priority);
            queuesize_++;
            heapify_up(priority);
            
        }

        
		// remove the top element from the pqueue
        sn_id_t
        pop()
        {
            if (queuesize_ == 0)
            {
                return SN_ID_MAX;
            }

            sn_id_t ans = elts_[0];
            queuesize_--;

            if(queuesize_ > 0)
            {
                elts_[0] = elts_[queuesize_];
                priority_[elts_[0]] = 0;
                heapify_down(0);
            }
            
            return ans;
        }

        
		// @return true if the priority of the element is 
		// otherwise
		inline bool
		contains(sn_id_t vertex)
		{
			unsigned int priority = priority_[vertex];
			if(priority < queuesize_ && vertex == elts_[priority] )
			{
				return true;
			}
			return false;
		}

		// retrieve the top element without removing it
		inline sn_id_t 
		peek()
		{
			if(queuesize_ > 0)
			{
				return this->elts_[0];

			}
			return 0;
		}

        
		inline unsigned int
		size() const
		{
			return queuesize_;
		}

        inline cost_t**
		get_cost_array() const
		{
			return cost_;
		}

        inline void
		set_cost_array(cost_t** cost_array_)
		{
			cost_ = cost_array_;
		}

        inline void
		set_cost_index(size_t cost_index)
		{
			cost_index_ = cost_index;
		}

		// inline bool
		// is_minqueue() 
		// { 
		// 	return minqueue_; 
		// } 
		
        // size_t
		// mem()
		// {
		// 	return maxsize_*sizeof(warthog::search_node*)
		// 		+ sizeof(*this);
		// }

        // inline void
        // set_index(bool index)
        // {index_ = index;}

	private:
		unsigned int initsize_;
		unsigned int maxsize_;
		// bool minqueue_;
		unsigned int queuesize_;
        unsigned int *priority_;
		sn_id_t *elts_;
        cost_t **cost_;
        size_t cost_index_;
        // Comparator* cmp_;
        
        // bool index_;
        
        

		// reorders the subpqueue containing elts_[index]
        void 
        heapify_up(sn_id_t index)
        {
            // assert(index < queuesize_);
            
            while(index > 0)
            {
                sn_id_t parent = (index-1) >> 1; //Shift right dividing by 2
                
                if(cost_[elts_[index]][cost_index_] < cost_[elts_[parent]][cost_index_])
                {
                    swap(parent, index);
                    index = parent;
                }
                else { break; }
            }
        }

    		
		// reorders the subpqueue under elts_[index]
        void 
        heapify_down(unsigned int index)
        {
            unsigned int first_leaf_index = queuesize_ >> 1;
            while(index < first_leaf_index)
            {
                // find smallest (or largest, depending on heap type) child
                unsigned int child1 = (index<<1)+1;
                unsigned int child2 = (index<<1)+2;
                unsigned int which = child1;
                if( child2 < queuesize_ && cost_[elts_[child2]][cost_index_] < cost_[elts_[child1]][cost_index_] ) 
                { which = child2; }

                // swap child with parent if necessary
                if(cost_[elts_[which]][cost_index_] < cost_[elts_[index]][cost_index_])
                {
                    swap(index, which);
                    index = which;
                }
                else { break; }
            }
        }

        // allocates more memory so the pqueue can grow
        void
        resize(unsigned int newsize)
        {
            // if(newsize < queuesize_)
            // {
            //     exit(1);
            // }

            sn_id_t* tmp = new sn_id_t[newsize];
            for(unsigned int i=0; i < queuesize_; i++)
            {
                tmp[i] = elts_[i];
            }
            delete [] elts_;
            elts_ = tmp;
            maxsize_ = newsize;
        }

        // swap the positions of two nodes in the underlying array
		inline void 
		swap(unsigned int index1, unsigned int index2)
		{
			// assert(index1 < queuesize_ && index2 < queuesize_);
            sn_id_t tmp1 = elts_[index1];
            elts_[index1] = elts_[index2];
            elts_[index2] = tmp1;
            priority_[elts_[index1]] = index1;
            priority_[elts_[index2]] = index2;
            
		}
};
///////////////////////////////////////////////
// PQUEUE MAX
class pqueue_max 
{
    
	public:
    
        pqueue_max(unsigned int size = 1024, cost_t* cost_array = 0, sn_id_t num_nodes = 0)
            : initsize_(size), maxsize_(size), queuesize_(0), elts_(0)
        {
            resize(size);
            priority_ = new unsigned int [num_nodes]();
            cost_ = cost_array;
        }

        ~pqueue_max()
        {
            delete [] elts_;
            delete [] priority_;
        }

        // removes all elements from the pqueue
        void
        clear()
        {
            queuesize_ = 0;
            maxsize_ = initsize_;
        }

		// reprioritise the specified element (up or down)
        void 
        increase_key(sn_id_t vertex, cost_t  f_org = 0)
        {	
            heapify_up(priority_[vertex]);
        }

        void 
        decrease_key(sn_id_t vertex)
        {
            heapify_down(priority_[vertex]);
        }

		// add a new element to the pqueue
        void 
        push(sn_id_t vertex)
        {
            // if(contains(vertex))
            // {
            //     return;
            // }
            
            if(queuesize_+1 > maxsize_)
            {
                resize(maxsize_*2);
            }
            unsigned int priority = queuesize_;
            elts_[priority] = vertex;
            priority_[vertex] = priority;
            
            // val->set_priority(index_,priority);
            queuesize_++;
            heapify_up(priority);
            
        }

        
		// remove the top element from the pqueue
        sn_id_t
        pop()
        {
            if (queuesize_ == 0)
            {
                return SN_ID_MAX;
            }

            sn_id_t ans = elts_[0];
            queuesize_--;

            if(queuesize_ > 0)
            {
                elts_[0] = elts_[queuesize_];
                priority_[elts_[0]] = 0;
                heapify_down(0);
            }
            
            return ans;
        }

        
		// @return true if the priority of the element is 
		// otherwise
		inline bool
		contains(sn_id_t vertex)
		{
			unsigned int priority = priority_[vertex];
			if(priority < queuesize_ && vertex == elts_[priority] )
			{
				return true;
			}
			return false;
		}

		// retrieve the top element without removing it
		inline sn_id_t 
		peek()
		{
			if(queuesize_ > 0)
			{
				return this->elts_[0];

			}
			return 0;
		}

        
		inline unsigned int
		size() const
		{
			return queuesize_;
		}

        inline cost_t*
		get_cost_array() const
		{
			return cost_;
		}

        inline void
		set_cost_array(cost_t* cost_array_)
		{
			cost_ = cost_array_;
		}

		// inline bool
		// is_minqueue() 
		// { 
		// 	return minqueue_; 
		// } 
		
        // size_t
		// mem()
		// {
		// 	return maxsize_*sizeof(warthog::search_node*)
		// 		+ sizeof(*this);
		// }

        // inline void
        // set_index(bool index)
        // {index_ = index;}

	private:
		unsigned int initsize_;
		unsigned int maxsize_;
		// bool minqueue_;
		unsigned int queuesize_;
        unsigned int *priority_;
		sn_id_t *elts_;
        cost_t *cost_;
        // Comparator* cmp_;
        
        // bool index_;
        
        

		// reorders the subpqueue containing elts_[index]
        void 
        heapify_up(sn_id_t index)
        {
            // assert(index < queuesize_);
            
            while(index > 0)
            {
                sn_id_t parent = (index-1) >> 1; //Shift right dividing by 2
                
                if(cost_[elts_[index]] > cost_[elts_[parent]])
                {
                    swap(parent, index);
                    index = parent;
                }
                else { break; }
            }
        }

    		
		// reorders the subpqueue under elts_[index]
        void 
        heapify_down(unsigned int index)
        {
            unsigned int first_leaf_index = queuesize_ >> 1;
            while(index < first_leaf_index)
            {
                // find smallest (or largest, depending on heap type) child
                unsigned int child1 = (index<<1)+1;
                unsigned int child2 = (index<<1)+2;
                unsigned int which = child1;
                if( child2 < queuesize_ && cost_[elts_[child2]] > cost_[elts_[child1]] ) 
                { which = child2; }

                // swap child with parent if necessary
                if(cost_[elts_[which]] > cost_[elts_[index]])
                {
                    swap(index, which);
                    index = which;
                }
                else { break; }
            }
        }

        // allocates more memory so the pqueue can grow
        void
        resize(unsigned int newsize)
        {
            // if(newsize < queuesize_)
            // {
            //     exit(1);
            // }

            sn_id_t* tmp = new sn_id_t[newsize];
            for(unsigned int i=0; i < queuesize_; i++)
            {
                tmp[i] = elts_[i];
            }
            delete [] elts_;
            elts_ = tmp;
            maxsize_ = newsize;
        }

        // swap the positions of two nodes in the underlying array
		inline void 
		swap(unsigned int index1, unsigned int index2)
		{
			// assert(index1 < queuesize_ && index2 < queuesize_);
            sn_id_t tmp1 = elts_[index1];
            elts_[index1] = elts_[index2];
            elts_[index2] = tmp1;
            priority_[elts_[index1]] = index1;
            priority_[elts_[index2]] = index2;
            
		}
};
#endif

