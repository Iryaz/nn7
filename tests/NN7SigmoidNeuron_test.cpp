
#include <iostream>
#include "nn7sigmoidneuron.h"

#define CATCH_CONFIG_MAIN
#include <catch.hpp>

#define ALPHA 1.0f

using std::cout;

struct TEST_CASE_obj {
  double center;
  double response;
  NN7SigmoidNeuron *sigmaNeuron;
};

TEST_CASE("Welcome Sigmoid neuron test case") {

  TEST_CASE_obj tests[2];
  double testVectors[3][3] = {
    {1.5, 1.1, 2},    // Positive vector
    {0.0, 0.0, 0.0},  // Null vector
    {-1.5, -1.1, -2}  // Negative vector
  };

  NN7Randomizer gen(0, 1);
  tests[0].center = 0.5f;
  tests[0].sigmaNeuron = new NN7UnipolarSigmoidNeuron(3, &gen, ALPHA);

  tests[1].center = 0.0f;
  tests[1].sigmaNeuron = new NN7BipolarSigmoidNeuron(3, &gen, ALPHA);

  cout << "\nBegin tests sigma neuron \n";
  cout << "    (0) - unipolar neuron\n";
  cout << "    (1) - bipolar neuron\n\n";

  for (int i = 0; i < 2; i++)
  {
    for (int j = 0; j < 3; j++) {
      NN7DataVector v(&testVectors[j][0], 3);
      tests[i].response = tests[i].sigmaNeuron->response(&v);

      switch(j)
      {
        case 0: {
          cout << "Response neuron (" << i << ") - " << tests[i].response << "\n";
          REQUIRE(tests[i].response > tests[i].center);
        }
        break;
        case 1: {
          cout << "Response neuron (" << i << ") - " << tests[i].response << "\n";
          REQUIRE(tests[i].response == tests[i].center);
        }
        break;
        case 2: {
          cout << "Response neuron (" << i << ") - " << tests[i].response << "\n";
          REQUIRE(tests[i].response < tests[i].center);
        }
        break;
      }
    }

    cout << "\n";
  }

  delete tests[0].sigmaNeuron;
  delete tests[1].sigmaNeuron;
}
