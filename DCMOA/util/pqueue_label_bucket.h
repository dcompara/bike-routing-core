#ifndef PQUEUE_LABEL_BUCKET_H
#define PQUEUE_LABEL_BUCKET_H

// A bucket-based priority queue for search_labels. 
// WITHOUT tie breaking using linked lists inside buckets
// First implementation pqueue_label_bucket: Fixed range buckets (needs f_max-f_min)
// Second implementation pqueue_label_bucket_cyclic: Cyclic buckets (needs bucket_size)
// Third implementation pqueue_label_bucket_cyclic_fixed: Cyclic fixed-sized buckets 
//
// 
//
// @author: sahmadi
// @created: 04/06/2021
// @updated: 28/01/2022
//

#include <cassert>
#include <iostream>
#include "labels.h"

// A simple single-layer bucket queue with Last In First Out (LIFO) strategy
template <class LABEL>
class pqueue_label_bucket
{
    
	public:
    
        pqueue_label_bucket(uint size=1, cost_t fmin=0, cost_t fmax=0)
            : max_cursor_(0), queuesize_(0), cursor_(0), f_min(fmin)
        {
            elts_ = new linkedlist<LABEL*>[1]();
            bucket_width_ = 1;
            fmax = (fmax > fmin) ? fmax : fmin +  1;
            initsize_ = (fmax - fmin)/bucket_width_ + 1;
            resize(initsize_);
        }

        ~pqueue_label_bucket()
        {
            delete [] elts_;
        }

        
        inline void
        range_update(cost_t fmin, cost_t fmax)
        {
            f_min = fmin;
            fmax = (fmax > fmin) ? fmax : fmin +  1;
            initsize_ = (fmax - fmin)/bucket_width_ + 1;
            resize(initsize_);
        }

        void
        clear()
        {
            queuesize_ = 0;
            max_cursor_ = 0;
            delete [] elts_;
            elts_ = new linkedlist<LABEL*>[1]();
            cursor_ = 0;
        }

		// add a new element to the pqueue
        void 
        push (LABEL* lb)
        {
            uint index = (lb->get_f_pri() - f_min);
            
            if (index > max_cursor_) resize(index*1.5);
            // if (index < cursor_ || index > max_cursor_)
            // std::cerr<<"Error: Accessing out of Bucket list."<<std::endl;
            // LIFO
            (this->elts_[index]).push_front(lb);
            // FIFO
            // (this->elts_[index]).push_back(lb);

            queuesize_++;
            // std::cerr<<"Error: Accessing out of Bucket list."<<std::endl;
            return;
        }

		LABEL*
        pop()
        {
            if (queuesize_)
            {
                LABEL* ans = this->peek();
                (this->elts_[cursor_]).pop_front();
                queuesize_--;
                return ans;
            }
            return 0;
        }

		
		inline LABEL*
		peek()
		{
            while(!elts_[cursor_].size())
                {
                    cursor_++;
                    // cmp++;
                }
                
            return this->elts_[cursor_].front();
        }

        inline cost_t 
		peek_f(void)
		{
            LABEL* ans = this->peek();
            
            if(ans)
			{
                return this->cursor_+this->f_min;
            }
            // std::cerr<<"Error: Accessing out of Bucket list."<<std::endl;
            return COST_MAX;
        }

        inline cost_t 
		current_f(void)
		{
            return this->cursor_*bucket_width_ + this->f_min;
        }

		inline uint
		size() const
		{
			return queuesize_;
		}

        void
        resize(uint newsize)
        {
            if(newsize - 1 <= max_cursor_)
            {return;}

            linkedlist<LABEL*>* tmp_ = new linkedlist<LABEL*>[newsize]();

            for(uint i=0; i <= max_cursor_; i++)
	        {
                tmp_[i] = elts_[i];
            }
            
            delete [] elts_;
            elts_ = tmp_;
            max_cursor_ = newsize-1;
            return;
        }

        inline size_t
        get_cmp()
        {
            return cursor_;
        }

        void 
        pull_up(LABEL* lb, LABEL* lb_old)
        {
            lb_old->set_f_pri(0);
            this->push(lb);
            return;
        
        }


        void 
        pull_down(LABEL* lb)
        {
            return;
        }
        
        size_t
		mem()
		{
			return max_cursor_*sizeof (LABEL*)
				+ sizeof(*this);
		}

    private:
		uint initsize_;
		uint max_cursor_;
		uint queuesize_;
		uint cursor_;
        cost_t f_min;
        uint bucket_width_;
        linkedlist<LABEL*>* elts_;
        // size_t cmp;
    
};

typedef pqueue_label_bucket<search_label> pqueue_label_bucket_min;
typedef pqueue_label_bucket<search_label_light> pqueue_label_light_bucket_min;

//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

// A simple single-layer bucket queue with First In First Out (FIFO) strategy
template <class LABEL>
class pqueue_label_bucket_fifo
{
    
	public:
    
        pqueue_label_bucket_fifo(uint size=1, cost_t fmin=0, cost_t fmax=0)
            : max_cursor_(0), queuesize_(0), cursor_(0), f_min(fmin)
        {
            elts_ = new linkedlist<LABEL*>[1]();
            bucket_width_ = 1;
            fmax = (fmax > fmin) ? fmax : fmin +  1;
            initsize_ = (fmax - fmin)/bucket_width_ + 1;
            resize(initsize_);
        }

        ~pqueue_label_bucket_fifo()
        {
            delete [] elts_;
        }

        
        inline void
        range_update(cost_t fmin, cost_t fmax)
        {
            f_min = fmin;
            fmax = (fmax > fmin) ? fmax : fmin +  1;
            initsize_ = (fmax - fmin)/bucket_width_ + 1;
            resize(initsize_);
        }
        

        void
        clear()
        {
            queuesize_ = 0;
            max_cursor_ = initsize_;
            delete [] elts_;
            elts_ = 0;
            cursor_ = 0;
        }

		// add a new element to the pqueue
        void 
        push (LABEL* lb)
        {
            uint index = (lb->get_f_pri() - f_min);
            
            if (index > max_cursor_) resize(index*1.5);

            // LIFO
            // (this->elts_[index]).push_front(lb);
            // FIFO
            (this->elts_[index]).push_back(lb);

            if (index < min_label_cursor_) min_label_cursor_ = index;
            
            queuesize_++;
            // if (index > max_cursor_)
            // std::cerr<<"Error: Accessing out of Bucket list."<<std::endl;
            return;
        }

		LABEL*
        pop()
        {
            if (queuesize_)
            {
                LABEL* ans = this->peek();
                (this->elts_[cursor_]).pop_front();
                queuesize_--;
                return ans;
            }
            return 0;
        }

		
		inline LABEL*
		peek()
		{
            while(!elts_[cursor_].size())
                {cursor_++;}
                
            return this->elts_[cursor_].front();
        }

        LABEL*
        peek_min_label()
        {
            if (queuesize_)
            {
                while(!elts_[min_label_cursor_].front())
                {min_label_cursor_++;}
                return elts_[min_label_cursor_].front();
            }
            return 0;
        }

        inline cost_t 
		peek_f(void)
		{
            LABEL* ans = this->peek();
            
            if(ans)
			{
                return this->cursor_+this->f_min;
            }
            // std::cerr<<"Error: Accessing out of Bucket list."<<std::endl;
            return COST_MAX;
        }

        inline cost_t 
		current_f(void)
		{
            return this->cursor_*bucket_width_ + this->f_min;
        }

		inline uint
		size() const
		{
			return queuesize_;
		}

        void
        resize(uint newsize)
        {
            if(newsize - 1 <= max_cursor_)
            {return;}

            linkedlist<LABEL*>* tmp_ = new linkedlist<LABEL*>[newsize]();

            for(uint i=0; i <= max_cursor_; i++)
	        {
                tmp_[i] = elts_[i];
            }
            
            delete [] elts_;
            elts_ = tmp_;
            max_cursor_ = newsize-1;
            return;
        }

        inline size_t
        get_cmp()
        {
            return cursor_;
        }

        void 
        pull_up(LABEL* lb, LABEL* lb_old)
        {
            lb_old->set_f_pri(0);
            this->push(lb);
            return;
        
        }


        void 
        pull_down(LABEL* lb)
        {
            return;
        }
        
        size_t
		mem()
		{
			return max_cursor_*sizeof (LABEL*)
				+ sizeof(*this);
		}

    private:
		uint initsize_;
		uint max_cursor_;
		uint queuesize_;
		uint cursor_;
        cost_t f_min;
        uint bucket_width_;
        linkedlist<LABEL*>* elts_;
        uint min_label_cursor_ = UINT_MAX;
    
};

typedef pqueue_label_bucket_fifo<search_label> pqueue_label_bucket_fifo_min;
typedef pqueue_label_bucket_fifo<search_label_light> pqueue_label_light_bucket_fifo_min;

//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
// A simple single-layer cyclic bucket queue with Last In First Out (LIFO) strategy
// The bucket list may be resized depending on the f-value
// Can be used for scenarions with unknown delta_f value
template <class LABEL>
class pqueue_label_bucket_cyclic
{
    
	public:
    
        pqueue_label_bucket_cyclic(uint size=1, cost_t fmin=0, cost_t fmax=0)
            : max_cursor_(1), queuesize_(0), cursor_(0), f_min(fmin), offset_(0)
        {
            bucket_size_ = size;
            offset_ = 0;
            elts_ = new linkedlist<LABEL*>[1];
        }

        ~pqueue_label_bucket_cyclic()
        {
            delete [] elts_;
        }

        void
        clear()
        {
            queuesize_ = 0;
            max_cursor_ = 1;
            delete [] elts_;
            elts_ = new linkedlist<LABEL*>[1]();
            cursor_ = 0;
        }

        inline void
        range_update(cost_t fmin, cost_t fmax)
        {
            f_min = fmin;
            // resize(fmax-fmin+1);
        }

		// add a new element to the pqueue
        void 
        push (LABEL* lb)
        {
            uint index = (lb->get_f_pri() - f_min)/bucket_size_ ;
            if (index > max_cursor_)
            {
                resize(index);
                index -= offset_;
                min_label_cursor_ -= offset_;
                offset_ = 0;
                cursor_ = 0;
            }

            if (index < min_label_cursor_) min_label_cursor_ = index;

            // LIFO
            (this->elts_[(index + offset_) % (max_cursor_)]).push_front(lb);
            // FIFO
            // (this->elts_[(index + offset_) % (max_cursor_)]).push_back(lb);
            queuesize_++;
            return;
        }

		LABEL*
        pop()
        {
            if (queuesize_)
            {
                LABEL* ans = this->peek();
                (this->elts_[cursor_ % max_cursor_]).pop_front();
                queuesize_--;
                return ans;
            }
            return 0;
        }

		
		inline LABEL*
		peek()
		{
            if(queuesize_)
			{
                while(!elts_[cursor_ % max_cursor_].front())
                    {
                        cursor_++;
                        offset_++;
                        f_min += bucket_size_ ;
                    }
                return this->elts_[cursor_ % max_cursor_].front();
            }
            return 0;
        }

        inline cost_t 
		peek_f(void)
		{
            
            LABEL* ans = this->peek();
            
            if(ans)
			{
                return (this->cursor_*bucket_size_) + this->f_min;
            }
            // std::cerr<<"Error: Accessing out of Bucket list.- Peek_f"<<std::endl;
            return COST_MAX;
        }

        LABEL*
        peek_min_label()
        {
            if (queuesize_)
            {
                while(!elts_[min_label_cursor_ % max_cursor_].front())
                {min_label_cursor_++;}
                return elts_[min_label_cursor_ % max_cursor_].front();
            }
            return 0;
        }

        inline cost_t 
		current_f(void)
		{
            return (this->cursor_*bucket_size_) + this->f_min;
        }

		inline uint
		size() const
		{
			return queuesize_;
		}

        void
        resize(uint newsize)
        {
            if(newsize <= max_cursor_)
            {return;}

            linkedlist<LABEL*>* tmp_ = new linkedlist<LABEL*>[newsize]();
            for(uint i=0; i < max_cursor_; i++)
	        {
                tmp_[i] = elts_[(i + offset_) % max_cursor_];
            }
            delete [] elts_;
            elts_ = tmp_;
            max_cursor_ = newsize;
            return;
        }

        inline size_t
        get_cmp()
        {
            return cursor_;
        }

        size_t
		mem()
		{
			return max_cursor_*sizeof (LABEL*)
				+ sizeof(*this);
		}

    private:
		uint initsize_;
		uint max_cursor_;
		uint queuesize_;
		uint cursor_;
        cost_t f_min;
        uint offset_;
        uint bucket_size_;
        linkedlist<LABEL*>* elts_;
        uint min_label_cursor_ = UINT_MAX;
    
};

typedef pqueue_label_bucket_cyclic<search_label> pqueue_label_bucket_cyclic_min;
typedef pqueue_label_bucket_cyclic<search_label_light> pqueue_label_light_bucket_cyclic_min;

///////////////////////////////////////////////////////////////////
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
// A simple single-layer fixed-size cyclic bucket queue with Last In First Out (LIFO) strategy
// The size of the bucket list is determined by delat_f = f_max - f_min
template <class LABEL>
class pqueue_label_bucket_cyclic_fixed
{
    
	public:
    
        pqueue_label_bucket_cyclic_fixed(uint size=1, cost_t fmin=0, cost_t max_delta_f=0)
            : max_cursor_(max_delta_f), queuesize_(0), cursor_(0), f_min(fmin), offset_(0), cmp_(0)
        {
            bucket_size_ = size;
            offset_ = 0;
            elts_ = new linkedlist<LABEL*>[max_delta_f + 1];
        }

        ~pqueue_label_bucket_cyclic_fixed()
        {
            delete [] elts_;
        }

        void
        clear()
        {
            queuesize_ = 0;
            max_cursor_ = 1;
            delete [] elts_;
            elts_ = new linkedlist<LABEL*>[1]();
            cursor_ = 0;
        }

        inline void
        range_update(cost_t fmin, cost_t fmax)
        {
            f_min = fmin;
            resize(fmax + 1);
        }

		// add a new element to the pqueue
        void 
        push (LABEL* lb)
        {
            uint index = (lb->get_f_pri() - f_min)/bucket_size_ ;
            
            // LIFO
            (this->elts_[(index + offset_) % (max_cursor_)]).push_front(lb);
            // FIFO
            // (this->elts_[(index + offset_) % (max_cursor_)]).push_back(lb);
            queuesize_++;
            return;
        }

		LABEL*
        pop()
        {
            if (queuesize_)
            {
                LABEL* ans = this->peek();
                (this->elts_[cursor_ % max_cursor_]).pop_front();
                queuesize_--;
                return ans;
            }
            return 0;
        }

		
		inline LABEL*
		peek()
		{
            if(queuesize_)
			{
                
                while(!elts_[cursor_ % max_cursor_].front())
                    {
                        // cmp_++;
                        cursor_++;
                        offset_++;
                        f_min += bucket_size_ ;
                    }
                return this->elts_[cursor_ % max_cursor_].front();
            }
            return 0;
        }

        inline cost_t 
		peek_f(void)
		{
            
            LABEL* ans = this->peek();
            
            if(ans)
			{
                return this->f_min; 
            }
            // std::cerr<<"Error: Accessing out of Bucket list.- Peek_f"<<std::endl;
            return COST_MAX;
        }

        LABEL*
        peek_min_label()
        {
            if (queuesize_)
            {
                while(!elts_[min_label_cursor_ % max_cursor_].front())
                {min_label_cursor_++;}
                return elts_[min_label_cursor_ % max_cursor_].front();
            }
            return 0;
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

        void
        resize(uint newsize)
        {
            if(newsize <= max_cursor_)
            {return;}

            linkedlist<LABEL*>* tmp_ = new linkedlist<LABEL*>[newsize]();
            for(uint i=0; i < max_cursor_; i++)
	        {
                tmp_[i] = elts_[(i + offset_) % max_cursor_];
            }
            delete [] elts_;
            elts_ = tmp_;
            max_cursor_ = newsize;
            return;
        }

        inline size_t
        get_cmp()
        {
            return cmp_;
        }

        size_t
		mem()
		{
			return max_cursor_*sizeof (LABEL*)
				+ sizeof(*this);
		}

    private:
		uint initsize_;
		uint max_cursor_;
		uint queuesize_;
		uint cursor_;
        cost_t f_min;
        uint offset_;
        uint bucket_size_;
        linkedlist<LABEL*>* elts_;
        uint min_label_cursor_ = UINT_MAX;
        size_t cmp_;
    
};

#endif