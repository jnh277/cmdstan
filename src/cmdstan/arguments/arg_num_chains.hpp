#ifndef CMDSTAN_ARGUMENTS_ARG_NUM_CHAINS_HPP
#define CMDSTAN_ARGUMENTS_ARG_NUM_CHAINS_HPP

#include <cmdstan/arguments/singleton_argument.hpp>

namespace cmdstan {

class arg_num_chains : public int_argument {
 public:
  arg_num_chains() : int_argument() {
    _name = "num_chains";
    _description = std::string("Number of chains");
    _default = "1";
    _default_value = 1;
    _value = _default_value;
  }

  bool is_valid(int value) { return value > 0; }
};

}  // namespace cmdstan
#endif
