#include <getopt.h>
#include <map>

// memory/label_pool.h
//
// A memory pool of search labels.
//
// @author: sahmadi
// @created: 04/06/2021
//

void parse_args(int argc, char **argv, const char *options,
                const option params[], experiment &exp)
{
    std::string::size_type sz;
    uint32_t id_offset = 0;
    int current_opt;
    for (int c = getopt_long(argc, argv, options, params, &current_opt);
         c != -1;
         c = getopt_long(argc, argv, options, params, &current_opt))
    {
        if (c == '?')
            break;
        else if (c == 'a')
            exp.alg_name = optarg;
        else if (c == 'i')
            {
                exp.file_name = optarg;
                std::string::size_type const p(exp.file_name.find_last_of('.'));
                std::string file_without_extension = exp.file_name.substr(0, p);
                exp.map_name = file_without_extension.substr(exp.file_name.find_last_of("/\\") + 1);
            }
        else if (c == 'q')
            {exp.queue_type = optarg;}
        else if (c == 's')
            exp.start = std::stoi(optarg, &sz) - id_offset;
        else if (c == 'g')
            exp.goal = std::stoi(optarg, &sz) - id_offset;
        else if (c == 'c')
            {
                exp.constraint = std::stoi(optarg, &sz);
                exp.constraint = exp.constraint > 100 ? 100 : exp.constraint;
            }
            
        else if (c == 'p')
            exp.path = 1;
        else if (c == 'r')
            exp.reverse = 1;
        else if (c == 'n')
            exp.nruns = std::stoi(optarg, &sz);
    }
}
