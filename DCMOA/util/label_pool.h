#ifndef label_POOL_H
#define label_POOL_H

// memory/label_pool.h
//
// A memory pool of search labels.
//
// @author: sahmadi
// @created: 04/06/2021
//

namespace label_pool_ls
{
	static const uint64_t LBS = 1024 * 32; // label block size; set this >= 8
	static const uint64_t LOG2_LBS = 10 + 5;
}

template <class L>
class label_pool
{
public:
	label_pool(size_t num_labels = 1024)
		: blocks_(0), head_free_(0), block_id_(0), label_id_(0)
	{
		init(num_labels);
	}

	~label_pool()
	{
		for (size_t i = 0; i < num_blocks_; i++)
		{
			if (blocks_[i] != 0)
			{
				delete[] blocks_[i];
				blocks_[i] = 0;
			}
		}
		delete[] blocks_;
	}

	L *const
	get_label(void)
	{
		L *tmp = 0;
		// id outside the pool address range
		if (head_free_)
		{
			tmp = head_free_;
			head_free_ = tmp->get_next();
			// tmp->set_next(0);
			return tmp;
		}
		else
		{
			if (!blocks_[block_id_])
			{
				blocks_[block_id_] = new L[label_pool_ls::LBS]();
			}
			tmp = &blocks_[block_id_][label_id_];
			label_id_++;
			if (label_id_ == label_pool_ls::LBS)
			{
				block_id_++;
				label_id_ = 0;
				if (block_id_ == num_blocks_)
				{
					size_t new_size = std::round(num_blocks_ * 1.5);
					resize(new_size);
				}
			}
		}
		return tmp;
	}

	uint32_t
	size(void)
	{
		return (block_id_ * label_pool_ls::LBS) + label_id_;
	}

	void
	save_label(L *label)
	{
		label->set_next(head_free_);
		head_free_ = label;
		return;
	}

	void
	save_label_chain(L *label_first, L *label_end)
	{
		label_end->set_next(head_free_);
		head_free_ = label_first;
		return;
	}

	size_t
	mem()
		{
			size_t block_size = !blocks_[block_id_] ? sizeof(L)*((block_id_)*label_pool_ls::LBS)
												   : sizeof(L)*((block_id_ + 1)*label_pool_ls::LBS);
			size_t bytes = 
				// memory of all labels
				block_size
				// memory for blocks (pointers)
				+ num_blocks_*sizeof(L*)
				// misc
				+ sizeof(*this);
			return bytes;
		}

private:
	size_t num_blocks_;
	L **blocks_;
	L *head_free_;
	uint16_t block_id_;
	uint16_t label_id_;

	void
	init(size_t num_labels)
	{
		num_blocks_ = ((num_labels) >> label_pool_ls::LOG2_LBS) + 1;
		blocks_ = new L *[num_blocks_]();
		for (size_t i = 0; i < num_blocks_; i++)
		{
			blocks_[i] = 0;
		}
	}

	void
	resize(size_t new_block_size)
	{
		L **tmp = new L *[new_block_size]();
		for (size_t i = 0; i < num_blocks_; i++)
		{
			tmp[i] = blocks_[i];
		}

		delete[] blocks_;
		blocks_ = tmp;
		num_blocks_ = new_block_size;
	}
};
///////////////
template <class L>
class label_pool_new
{
public:
	label_pool_new(){}

	label_pool_new(size_t num_labels, size_t num_obj)
		: blocks_(0), head_free_(0), block_id_(0), label_id_(0), num_obj_(num_obj)
	{
		init(num_labels);
	}

	~label_pool_new()
	{
		for (size_t i = 0; i < num_blocks_; i++)
		{
			if (blocks_[i] != 0)
			{
				delete[] blocks_[i];
				blocks_[i] = 0;
			}
		}
		delete[] blocks_;
	}

	L *const
	get_label(void)
	{
		L *tmp = 0;
		// id outside the pool address range
		if (head_free_)
		{
			tmp = head_free_;
			head_free_ = tmp->get_next();
			// tmp->set_next(0);
			return tmp;
		}
		else
		{
			if (!blocks_[block_id_])
			{
				blocks_[block_id_] = new L[label_pool_ls::LBS]();
			}
			tmp = &blocks_[block_id_][label_id_];
			label_id_++;
			if (label_id_ == label_pool_ls::LBS)
			{
				block_id_++;
				label_id_ = 0;
				if (block_id_ == num_blocks_)
				{
					size_t new_size = std::round(num_blocks_ * 1.5);
					resize(new_size);
				}
			}
		}
		return tmp;
	}

	uint32_t
	size(void)
	{
		return (block_id_ * label_pool_ls::LBS) + label_id_;
	}

	void
	save_label(L *label)
	{
		label->set_next(head_free_);
		head_free_ = label;
		return;
	}

	void
	save_label_chain(L *label_first, L *label_end)
	{
		label_end->set_next(head_free_);
		head_free_ = label_first;
		return;
	}

	size_t
	mem()
		{
			size_t block_size = !blocks_[block_id_] ? sizeof(L)*((block_id_)*label_pool_ls::LBS)
												   : sizeof(L)*((block_id_ + 1)*label_pool_ls::LBS);
			size_t bytes = 
				// memory of all labels
				block_size
				// memory for blocks (pointers)
				+ num_blocks_*sizeof(L*)
				// misc
				+ sizeof(*this);
			return bytes;
		}

private:
	size_t num_blocks_;
	L **blocks_;
	L *head_free_;
	uint16_t block_id_;
	uint16_t label_id_;
	size_t num_obj_;

	void
	init(size_t num_labels)
	{
		num_blocks_ = ((num_labels) >> label_pool_ls::LOG2_LBS) + 1;
		blocks_ = new L *[num_blocks_]();
		for (size_t i = 0; i < num_blocks_; i++)
		{
			blocks_[i] = 0;
		}
	}

	void
	resize(size_t new_block_size)
	{
		L **tmp = new L *[new_block_size]();
		for (size_t i = 0; i < num_blocks_; i++)
		{
			tmp[i] = blocks_[i];
		}

		delete[] blocks_;
		blocks_ = tmp;
		num_blocks_ = new_block_size;
	}
};
#endif

// typedef label_pool<warthog::search_label> label_pool_type;
