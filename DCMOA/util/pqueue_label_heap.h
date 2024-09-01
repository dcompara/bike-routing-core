#ifndef PQUEUE_LABEL_HEAP_H
#define PQUEUE_LABEL_HEAP_H
// A heap priority queue for search labels. 
// First implementation pqueue_label_heap: binary heap queue WITHOUT tie breaking
// Second implementation pqueue_label_heap_tie: binary heap queue WITH tie breaking
// 
//
// @author: sahmadi
// @updated: 04/06/2021
//


template <class LABEL = search_label_pri>
class pqueue_label_heap 
{
    
	public:
    
        pqueue_label_heap(uint size=1024, cost_t  fmin=0, cost_t  fmax=0)
            : initsize_(size), maxsize_(size), queuesize_(0), elts_(0), f_min(fmin), cmp(0)
        {
            resize(size);
        }

        ~pqueue_label_heap()
        {
            if (elts_)
            delete [] elts_;          
        }

        inline void
        range_update(cost_t  fmin, cost_t  fmax)
        {
            f_min = fmin;
        }

        inline void
        size_update(uint size)
        {resize(size);}

        // removes all elements from the pqueue_label_heap
        void
        clear()
        {
            queuesize_ = 0;
            resize(initsize_);
        }

		// add a new element to the pqueue_label_heap
        void 
        push(LABEL* lb)
        {
            if(queuesize_ + 1 > maxsize_)
            {
                resize(maxsize_*2);
            }
            uint priority = queuesize_;
            elts_[priority] = lb;
            lb->set_priority(priority);
            queuesize_++;
            heapify_up(priority);
        }

        void 
        pull_up(LABEL* lb, LABEL* lb2 = 0)
        {
            uint priority = lb2->get_priority();
            lb->set_priority(priority);
            elts_[priority] = lb;
            heapify_up(priority);
        }

		
        LABEL*
        pop()
        {
            
            if (queuesize_ == 0)
            {
                return NULL;
            }

            LABEL* ans=elts_[0];
            queuesize_--;

            if(queuesize_ > 0)
            {
                elts_[0] = elts_[queuesize_];
                elts_[0]->set_priority(0);// = 0;
                heapify_down(0);
            }

            if (ans->get_f_pri() > f_min) f_min = ans->get_f_pri();
            
            return ans;
        }

        void 
        pull_down(LABEL* lb)
        {
            uint priority = lb->get_priority();
            elts_[priority] = lb;
            heapify_down(priority);
        }

		
		inline LABEL*
		peek()
		{
			if(queuesize_ > 0)
			{
				return this->elts_[0];
			}
			return NULL;
		}

        inline cost_t 
		current_f(void)
		{
            return this->f_min;
        }

		inline uint
		size() const
		{
			return queuesize_;
		}

		size_t
		mem()
		{
			return maxsize_*sizeof(LABEL*)
				+ sizeof(*this);
		}

        pqueue_label_heap<LABEL>& 
		operator=(pqueue_label_heap<LABEL>& other) 
        { 
            this->initsize_ = other.initsize_;
            this->maxsize_ = other.maxsize_;
            this->queuesize_ = other.queuesize_;
            this->elts_ = other.elts_;
            other.elts_ = 0;
            return *this; 
        }

        inline size_t
        get_cmp()
        {
            return cmp;
        }

        
	private:
		uint initsize_;
		uint maxsize_;
        uint queuesize_;
		LABEL** elts_;
        cost_t f_min;
        size_t cmp;
        
		// reorders the subpqueue_label_heap containing elts_[index]
        void 
        heapify_up(uint index)
        {
            // assert(index < queuesize_);
            while(index > 0)
            {
                uint parent = (index-1) >> 1; //Shift right dividing by 2
                
                // if((cmp_)(elts_[index], elts_[parent], comp))
                if(*elts_[index] < *elts_[parent])
                {
                    swap(parent, index);
                    index = parent;
                }
                else { break; }
            }
        }
		
		void 
        heapify_down(uint index)
        {
            uint first_leaf_index = queuesize_ >> 1;
            while(index < first_leaf_index)
            {
                // find smallest (or largest, depending on heap type) child
                uint child1 = (index<<1)+1;
                uint child2 = (index<<1)+2;
                uint which = child1;
                if( child2 < queuesize_ &&
                    // (cmp_)(elts_[child2], elts_[child1], comp)) 
                    *elts_[child2] < *elts_[child1]) 
                
                { which = child2; }

                // swap child with parent if necessary
                // if((cmp_)(elts_[which], elts_[index], comp))
                if( *elts_[which] < *elts_[index])
                {
                    swap(index, which);
                    index = which;
                }
                else { break; }
            }
        }

		// allocates more memory so the pqueue_label_heap can grow
        void
        resize(uint newsize)
        {
            if(newsize < queuesize_)
            {
                exit(1);
            }
            LABEL** tmp = new LABEL*[newsize];
            for(uint i=0; i < queuesize_; i++)
            {
                tmp[i] = elts_[i];
            }
            delete [] elts_;
            elts_ = tmp;
            maxsize_ = newsize;
        }

		// swap the positions of two nodes in the underlying array
		inline void 
		swap(uint index1, uint index2)
		{
			// cmp++;
            assert(index1 < queuesize_ && index2 < queuesize_);
			LABEL* tmp = elts_[index1];
            elts_[index1] = elts_[index2];
            elts_[index2] = tmp;
            elts_[index1]->set_priority(index1);// = index1;
            elts_[index2]->set_priority(index2);// = index2;

		}
 

};
typedef pqueue_label_heap<search_label_pri> pqueue_label_heap_min_wo_tie;
typedef pqueue_label_heap<search_label_light_pri> pqueue_label_light_heap_min_wo_tie;
#endif


#ifndef PQUEUE_LABEL_HEAP_TIE_H
#define PQUEUE_LABEL_HEAP_TIE_H

template <class LABEL = search_label_pri>
class pqueue_label_heap_tie 
{
    
	public:
    
        pqueue_label_heap_tie(uint size=1, cost_t  fmin=0, cost_t  fmax=0)
            : initsize_(size), maxsize_(size), queuesize_(0), elts_(0), f_min(fmin), cmp(0)
        {
            resize(size);
        }

        ~pqueue_label_heap_tie()
        {
            if (elts_)
            delete [] elts_;          
        }

        inline void
        range_update(cost_t  fmin, cost_t  fmax)
        {
            f_min = fmin;
        }

        inline void
        size_update(uint size)
        {resize(size);}

        // removes all elements from the pqueue_label_heap
        void
        clear()
        {
            queuesize_ = 0;
            resize(initsize_);
        }

		// add a new element to the pqueue_label_heap
        void 
        push(LABEL* lb, cost_t f_val = 0)
        {
            if(queuesize_ + 1 > maxsize_)
            {
                resize(maxsize_*2);
            }
            uint priority = queuesize_;
            elts_[priority] = lb;
            lb->set_priority(priority);
            queuesize_++;
            heapify_up(priority);
        }

        void 
        pull_up(LABEL* lb, LABEL* lb2 = 0)
        {
            uint priority = lb2->get_priority();
            lb->set_priority(priority);
            elts_[priority] = lb;
            heapify_up(priority);
        }

		
        LABEL*
        pop()
        {
            
            if (queuesize_ == 0)
            {
                return NULL;
            }

            LABEL* ans=elts_[0];
            queuesize_--;

            if(queuesize_ > 0)
            {
                elts_[0] = elts_[queuesize_];
                elts_[0]->set_priority(0);// = 0;
                heapify_down(0);
            }
            
            if (ans->get_f_pri() > f_min) f_min = ans->get_f_pri();

            return ans;
        }

        void 
        pull_down(LABEL* lb)
        {
            uint priority = lb->get_priority();
            elts_[priority] = lb;
            heapify_down(priority);
        }

		
		inline LABEL*
		peek()
		{
			if(queuesize_ > 0)
			{
				return this->elts_[0];
			}
			return NULL;
		}

        inline cost_t 
		current_f(void)
		{
            return this->f_min;
        }

		inline uint
		size() const
		{
			return queuesize_;
		}

		size_t
		mem()
		{
			return maxsize_*sizeof(LABEL*)
				+ sizeof(*this);
		}

        pqueue_label_heap_tie<LABEL>& 
		operator=(pqueue_label_heap_tie<LABEL>& other) 
        { 
            this->initsize_ = other.initsize_;
            this->maxsize_ = other.maxsize_;
            this->queuesize_ = other.queuesize_;
            this->elts_ = other.elts_;
            other.elts_ = 0;
            return *this; 
        }

        inline size_t
        get_cmp()
        {
            return cmp; // number of comparisons
        }

        
	private:
		uint initsize_;
		uint maxsize_;
        uint queuesize_;
		LABEL** elts_;
        cost_t f_min;
        size_t cmp; 
        
		// reorders the subpqueue_label_heap containing elts_[index]
        void 
        heapify_up(uint index)
        {
            // assert(index < queuesize_);
            while(index > 0)
            {
                uint parent = (index-1) >> 1; //Shift right dividing by 2
                
                // if((cmp_)(elts_[index], elts_[parent], comp))
                if(*elts_[index] <= *elts_[parent])
                {
                    swap(parent, index);
                    index = parent;
                }
                else { break; }
            }
        }
		
		void 
        heapify_down(uint index)
        {
            uint first_leaf_index = queuesize_ >> 1;
            while(index < first_leaf_index)
            {
                // find smallest (or largest, depending on heap type) child
                uint child1 = (index<<1)+1;
                uint child2 = (index<<1)+2;
                uint which = child1;
                if( child2 < queuesize_ &&
                    // (cmp_)(elts_[child2], elts_[child1], comp)) 
                    *elts_[child2] <= *elts_[child1])
                
                { which = child2; }

                // swap child with parent if necessary
                // if((cmp_)(elts_[which], elts_[index], comp))
                if(*elts_[which] <= *elts_[index])
                {
                    swap(index, which);
                    index = which;
                }
                else { break; }
            }
        }

		// allocates more memory so the pqueue_label_heap can grow
        void
        resize(uint newsize)
        {
            if(newsize < queuesize_)
            {
                exit(1);
            }
            LABEL** tmp = new LABEL*[newsize];
            for(uint i=0; i < queuesize_; i++)
            {
                tmp[i] = elts_[i];
            }
            delete [] elts_;
            elts_ = tmp;
            maxsize_ = newsize;
        }

		// swap the positions of two nodes in the underlying array
		inline void 
		swap(uint index1, uint index2)
		{
			// cmp++;
            assert(index1 < queuesize_ && index2 < queuesize_);
			LABEL* tmp = elts_[index1];
            elts_[index1] = elts_[index2];
            elts_[index2] = tmp;
            elts_[index1]->set_priority(index1);// = index1;
            elts_[index2]->set_priority(index2);// = index2;

		}
 

};

typedef pqueue_label_heap_tie<search_label_pri> pqueue_label_heap_min_w_tie;
typedef pqueue_label_heap_tie<search_label_light_pri> pqueue_label_light_heap_min_w_tie;

#endif


