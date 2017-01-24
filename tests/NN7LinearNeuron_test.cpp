
#include <iostream>
#include "nn7linearneuron.h"

#define CATCH_CONFIG_MAIN
#include <catch.hpp>

using std::cout;

TEST_CASE("Welcome Base neuron test case") {

  NN7Randomizer randomGenerator(-1, 1);
  NN7LinearNeuron bn(5, &randomGenerator);

  SECTION("Init weights") {
    cout << "\n";
    NN7LinearNeuron::print_weight(bn);
  }

  SECTION("Test response") {
    for (uint32_t i = 0; i < bn.getInputsNum(); i++) {
      bn.setWeight(1 / 5.0f, i);
    }

    NN7DataVector x {1, 2, 3, 4, 5};
    double response = bn.response(&x);
    REQUIRE((int)response == 3);
  }

  SECTION("Test exception Input weight") {
    try {
      for (uint32_t i = 0; i < bn.getInputsNum() + 2; i++) {
        bn.setWeight(1 / 5.0f, i);
      }
    }
    catch(NN7Exception e) {
      NN7Exception::printAllException(e);
      cout << "\nIncorrect len input weight vector\n";
    }
  }

  SECTION("Test exception input vector") {
    try {
      NN7DataVector x {1.0, 2.0, 36.0, 12.0, 34.0, 0.0};
      bn.response(&x);
    }
    catch(NN7Exception e) {
      NN7Exception::printAllException(e);
      cout << "\nIncorrect len input data vector\n";
    }
  }
}
