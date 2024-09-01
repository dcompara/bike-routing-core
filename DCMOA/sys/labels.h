#ifndef SEARCH_LABEL_H
#define SEARCH_LABEL_H
// labels.h
//
// Search labels of the code, DIM is the dimension of cost vectors
// @author: sahmadi
// @updated: 01/09/2023
//

///////////////////////////////////////
// Truncated Search LABEL
struct search_label_tr
{
    std::array<cost_t, DIM - 1> f_;
    
    search_label_tr() 
    {
		for (dim_t i = 0; i < DIM - 1; i++)
		{f_[i] = COST_MAX;}
	}

	search_label_tr(const std::array<cost_t, DIM> &f) 
	{
		for (dim_t i = 0; i < DIM - 1; i++)
		{f_[i] = f[i + 1];}
	}
	search_label_tr(const cost_t *f) 
	{
		for (dim_t i = 0; i < DIM - 1; i++)
		{f_[i] = f[i + 1];}
	}

	search_label_tr(const std::array<cost_t, DIM> &f, dim_t offset) 
	{
		for (dim_t i = 0; i < DIM - 1; i++)
		{f_[i] = f[(i + 1 + offset) % DIM];}
	}

	search_label_tr(const cost_t *f, dim_t offset) 
	{
		for (dim_t i = 0; i < DIM - 1; i++)
		{f_[i] = f[(i + 1 + offset) % DIM];}
	}

	inline std::array<cost_t, DIM - 1>
	get_f() const
	{return f_;}
	
	// IsDominatedBy
    bool 
	operator <<(const search_label_tr& other) const
    {
		for (dim_t i = 0; i < DIM - 1; ++i)
		{if (f_[i] > other.f_[i]) return false;}
		return true;
	}
	bool 
	dominates(const search_label_tr& other, dim_t start_index = 0) const
    {
		for (dim_t i = start_index; i < DIM - 1; ++i)
		{if (f_[i] > other.f_[i]) return false;}
		return true;
	}
	// IsLexicographicallySmallerThan
    bool 
	operator <=(const search_label_tr& other) const
    {
        for (dim_t i = 0; i < DIM - 1; ++i)
		{
			if (f_[i] < other.f_[i]) return true;
			if (f_[i] > other.f_[i]) return false;
		}
		return false;
    }

	bool 
	operator ==(const search_label_tr& other) const
    {
        for (dim_t i = 0; i < DIM - 1; ++i)
		{
			if (f_[i] != other.f_[i]) return false;
		}
		return true;
    }
	
	bool 
	is_lex_smaller_than(const search_label_tr& other, dim_t start_index = 0) const
    {
        for (dim_t i = start_index; i < DIM - 1; ++i)
		{
			if (f_[i] < other.f_[i]) return true;
			if (f_[i] > other.f_[i]) return false;
		}
		return false;
    }

};
typedef search_label_tr LABEL_TR;
typedef std::vector<LABEL_TR> LABEL_TR_list;
typedef typename std::vector<LABEL_TR>::iterator LABEL_TR_iter;
typedef typename std::vector<LABEL_TR>::reverse_iterator LABEL_TR_rev_iter;
///////////////////////////////////////
//// LIGHT SEARCH LABEL
class search_label_light
{
public:
	search_label_light()
	{
		id_ = SN_ID_MAX;
		next_ = 0;
	}

	~search_label_light()
	{}

	// Non-Lexicographically smaller
	bool operator<(const search_label_light& other) 
	{return f_[0] < other.f_[0];}
	
	// Lexicographically smaller
	bool 
	operator <<(const search_label_light& other) const
    {
		for (dim_t i = 0; i < DIM; ++i)
		{if (f_[i] > other.f_[i]) return false;}
		return true;
	}
	// IsLexicographicallySmallerThan
    bool 
	operator <=(const search_label_light& other) const
    {
        for (dim_t i = 0; i < DIM; ++i)
		{
			if (f_[i] < other.f_[i]) return true;
			if (f_[i] > other.f_[i]) return false;
		}
		return false;
    }

	search_label_light(const std::array<cost_t, DIM> &f, sn_id_t id) 
	{
		for (dim_t i = 0; i < DIM; i++) {f_[i] = f[i];}
		id_ = id;
	}
	search_label_light(const std::array<cost_t, DIM> &f, sn_id_t id, search_label_light *parent)
	{
		for (dim_t i = 0; i < DIM; i++) {f_[i] = f[i];} 
		id_ = id;
		next_ = parent;
	}
	search_label_light(const std::array<cost_t, DIM> &f, sn_id_t id, vertex_deg_t incoming_edge, path_arr_size index)
	{
		for (dim_t i = 0; i < DIM; i++) {f_[i] = f[i];} 
		id_ = id;
	}
	search_label_light(cost_t *f, sn_id_t id, vertex_deg_t incoming_edge, path_arr_size index)
	{
		for (dim_t i = 0; i < DIM; i++) {f_[i] = f[i];} 
		id_ = id;
	}

	inline void
	set_f(const std::array<cost_t, DIM> &f)
	{
		for (dim_t i = 0; i < DIM; i++) {f_[i] = f[i];}
	}
	
	inline void
	set_id(sn_id_t id)
	{id_ = id;}

	inline std::array<cost_t, DIM>
	get_f() const
	{return f_;}

	inline cost_t
	get_f_pri() const
	{return f_[0];}

	inline sn_id_t
	get_id() const
	{return id_;}

	inline vertex_deg_t
	get_incoming_edge()
	{return DEG_MAX;}

	inline path_arr_size
	get_path_id()
	{return PATH_ARR_SIZE_MAX;}

	inline search_label_light *
	get_parent()
	{return next_;}

	inline uint
	get_priority()
	{return 0;}

	inline void
	set_priority(uint priority)
	{}

	inline search_label_light *
	get_next()
	{return next_;}

	inline void
	set_next(search_label_light *next)
	{next_ = next;}

	size_t
	mem()
	{return sizeof(*this);}

	// inline cost_t
	// get_rolling_cost() const
	// {return 0;}

	// inline void
	// set_rolling_cost(cost_t rolling_cost_) 
	// {}

	// bool check_cycle = false;
	// bool check_cycle2 = false;

private:
	std::array<cost_t, DIM> f_;
	sn_id_t id_;
	search_label_light *next_ = 0; // for linked list
};
///////////////////////////////////////
//// LIGHT SEARCH LABEL WITH PRIORITY
class search_label_light_pri
{
public:
	search_label_light_pri()
	{
		id_ = SN_ID_MAX;
		next_ = 0;
	}

	~search_label_light_pri()
	{}

	// Non-Lexicographically smaller
	bool operator<(const search_label_light_pri& other) 
	{return f_[0] < other.f_[0];}
	
	// Lexicographically smaller
	bool 
	operator <<(const search_label_light_pri& other) const
    {
		for (dim_t i = 0; i < DIM; ++i)
		{if (f_[i] > other.f_[i]) return false;}
		return true;
	}
	// IsLexicographicallySmallerThan
    bool 
	operator <=(const search_label_light_pri& other) const
    {
        for (dim_t i = 0; i < DIM; ++i)
		{
			if (f_[i] < other.f_[i]) return true;
			if (f_[i] > other.f_[i]) return false;
		}
		return false;
    }

	search_label_light_pri(const std::array<cost_t, DIM> &f, sn_id_t id) 
	{
		for (dim_t i = 0; i < DIM; i++) {f_[i] = f[i];}
		id_ = id;
	}
	search_label_light_pri(const std::array<cost_t, DIM> &f, sn_id_t id, search_label_light_pri *parent)
	{
		for (dim_t i = 0; i < DIM; i++) {f_[i] = f[i];} 
		id_ = id;
		next_ = parent;
	}
	search_label_light_pri(const std::array<cost_t, DIM> &f, sn_id_t id, vertex_deg_t incoming_edge, path_arr_size index)
	{
		for (dim_t i = 0; i < DIM; i++) {f_[i] = f[i];} 
		id_ = id;
	}
	search_label_light_pri(cost_t *f, sn_id_t id, vertex_deg_t incoming_edge, path_arr_size index)
	{
		for (dim_t i = 0; i < DIM; i++) {f_[i] = f[i];} 
		id_ = id;
	}

	inline void
	set_f(const std::array<cost_t, DIM> &f)
	{
		for (dim_t i = 0; i < DIM; i++) {f_[i] = f[i];}
	}

	inline void
	set_id(sn_id_t id)
	{id_ = id;}

	inline std::array<cost_t, DIM>
	get_f() const
	{return f_;}

	inline cost_t
	get_f_pri() const
	{return f_[0];}

	inline sn_id_t
	get_id() const
	{return id_;}

	inline vertex_deg_t
	get_incoming_edge()
	{return DEG_MAX;}

	inline path_arr_size
	get_path_id()
	{return PATH_ARR_SIZE_MAX;}

	inline search_label_light_pri *
	get_parent()
	{return next_;}

	inline uint
	get_priority()
	{return priority_;}

	inline void
	set_priority(uint priority)
	{priority_ = priority;}

	inline search_label_light_pri *
	get_next()
	{return next_;}

	inline void
	set_next(search_label_light_pri *next)
	{next_ = next;}

	size_t
	mem()
	{return sizeof(*this);}

	// inline cost_t
	// get_rolling_cost() const
	// {return 0;}

	// inline void
	// set_rolling_cost(cost_t rolling_cost_) 
	// {}

	// bool check_cycle = false;
	// bool check_cycle2 = false;

private:
	std::array<cost_t, DIM> f_;
	sn_id_t id_;
	search_label_light_pri *next_ = 0; // for linked list
	uint priority_;
};
///////////////////////////////////////
//// SEARCH LABEL
class search_label
{
public:
	search_label()
	{
		id_ = SN_ID_MAX;
		next_ = 0;
	}

	~search_label()
	{}

	// Non-Lexicographically smaller
	bool operator<(const search_label& other) 
	{return f_[0] < other.f_[0];}
	
	// Lexicographically smaller
	bool 
	operator <<(const search_label& other) const
    {
		for (dim_t i = 0; i < DIM; ++i)
		{if (f_[i] > other.f_[i]) return false;}
		return true;
	}
	// IsLexicographicallySmallerThan
    bool 
	operator <=(const search_label& other) const
    {
        for (dim_t i = 0; i < DIM; ++i)
		{
			if (f_[i] < other.f_[i]) return true;
			if (f_[i] > other.f_[i]) return false;
		}
		return false;
    }

	search_label(const std::array<cost_t, DIM> &f, sn_id_t id) 
	{
		for (dim_t i = 0; i < DIM; i++) {f_[i] = f[i];}
		id_ = id;
	}
	search_label(const std::array<cost_t, DIM> &f, sn_id_t id, search_label *parent)
	{
		for (dim_t i = 0; i < DIM; i++) {f_[i] = f[i];} 
		id_ = id;
		next_ = parent;
	}
	search_label(const std::array<cost_t, DIM> &f, sn_id_t id, vertex_deg_t incoming_edge, path_arr_size index)
	{
		for (dim_t i = 0; i < DIM; i++) {f_[i] = f[i];} 
		id_ = id;
		incoming_edge_ = incoming_edge;
		path_id_ = index;
	}
	search_label(cost_t *f, sn_id_t id, vertex_deg_t incoming_edge, path_arr_size index)
	{
		for (dim_t i = 0; i < DIM; i++) {f_[i] = f[i];} 
		id_ = id;
		incoming_edge_ = incoming_edge;
		path_id_ = index;
	}

	inline void
	set_f(const std::array<cost_t, DIM> &f)
	{
		for (dim_t i = 0; i < DIM; i++) {f_[i] = f[i];}
	}

	inline void
	set_id(sn_id_t id)
	{id_ = id;}

	inline std::array<cost_t, DIM>
	get_f() const
	{return f_;}

	inline cost_t
	get_f_pri() const
	{return f_[0];}

	inline sn_id_t
	get_id() const
	{return id_;}

	inline vertex_deg_t
	get_incoming_edge()
	{return incoming_edge_;}

	inline path_arr_size
	get_path_id()
	{return path_id_;}

	inline search_label *
	get_parent()
	{return next_;}

	inline uint
	get_priority()
	{return 0;}

	inline void
	set_priority(uint priority)
	{}

	inline search_label *
	get_next()
	{return next_;}

	inline void
	set_next(search_label *next)
	{next_ = next;}

	size_t
	mem()
	{return sizeof(*this);}

	// inline cost_t
	// get_rolling_cost() const
	// {return rolling_cost;}

	// inline void
	// set_rolling_cost(cost_t rolling_cost_) 
	// {rolling_cost = rolling_cost_;}

	// bool check_cycle = false;
	// bool check_cycle2 = false;
private:
	std::array<cost_t, DIM> f_;
	sn_id_t id_;
	search_label *next_ = 0; // for linked list
	vertex_deg_t incoming_edge_;
	path_arr_size path_id_;
	// cost_t rolling_cost;
};
///////////////////////////////////////
//// SEARCH LABEL WITH PRIORITY
class search_label_pri
{
public:
	search_label_pri()
	{
		id_ = SN_ID_MAX;
		next_ = 0;
	}

	~search_label_pri()
	{}

	// Non-Lexicographically smaller
	bool operator<(const search_label_pri& other) 
	{return f_[0] < other.f_[0];}
	
	// Lexicographically smaller
	bool 
	operator <<(const search_label_pri& other) const
    {
		for (dim_t i = 0; i < DIM; ++i)
		{if (f_[i] > other.f_[i]) return false;}
		return true;
	}
	// IsLexicographicallySmallerThan
    bool 
	operator <=(const search_label_pri& other) const
    {
        for (dim_t i = 0; i < DIM; ++i)
		{
			if (f_[i] < other.f_[i]) return true;
			if (f_[i] > other.f_[i]) return false;
		}
		return false;
    }

	search_label_pri(const std::array<cost_t, DIM> &f, sn_id_t id) 
	{
		for (dim_t i = 0; i < DIM; i++) {f_[i] = f[i];}
		id_ = id;
	}
	search_label_pri(const std::array<cost_t, DIM> &f, sn_id_t id, search_label_pri *parent)
	{
		for (dim_t i = 0; i < DIM; i++) {f_[i] = f[i];} 
		id_ = id;
		next_ = parent;
	}
	search_label_pri(const std::array<cost_t, DIM> &f, sn_id_t id, vertex_deg_t incoming_edge, path_arr_size index)
	{
		for (dim_t i = 0; i < DIM; i++) {f_[i] = f[i];} 
		id_ = id;
		incoming_edge_ = incoming_edge;
		path_id_ = index;
	}
	search_label_pri(cost_t *f, sn_id_t id, vertex_deg_t incoming_edge, path_arr_size index)
	{
		for (dim_t i = 0; i < DIM; i++) {f_[i] = f[i];} 
		id_ = id;
		incoming_edge_ = incoming_edge;
		path_id_ = index;
	}

	inline void
	set_f(const std::array<cost_t, DIM> &f)
	{
		for (dim_t i = 0; i < DIM; i++) {f_[i] = f[i];}
	}

	inline void
	set_id(sn_id_t id)
	{id_ = id;}

	inline std::array<cost_t, DIM>
	get_f() const
	{return f_;}

	inline cost_t
	get_f_pri() const
	{return f_[0];}

	inline sn_id_t
	get_id() const
	{return id_;}

	inline vertex_deg_t
	get_incoming_edge()
	{return incoming_edge_;}

	inline path_arr_size
	get_path_id()
	{return path_id_;}

	inline search_label_pri *
	get_parent()
	{return next_;}

	inline uint
	get_priority()
	{return priority_;}

	inline void
	set_priority(uint priority)
	{priority_ = priority;}

	inline search_label_pri *
	get_next()
	{return next_;}

	inline void
	set_next(search_label_pri *next)
	{next_ = next;}

	size_t
	mem()
	{return sizeof(*this);}

	// inline cost_t
	// get_rolling_cost() const
	// {return 0;}

	// inline void
	// set_rolling_cost(cost_t rolling_cost_) 
	// {}

	// bool check_cycle = false;
	// bool check_cycle2 = false;

private:
	std::array<cost_t, DIM> f_;
	sn_id_t id_;
	search_label_pri *next_ = 0; // for linked list
	vertex_deg_t incoming_edge_;
	path_arr_size path_id_;
	uint priority_;
};
#endif