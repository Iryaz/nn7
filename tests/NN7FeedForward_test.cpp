
#include <iostream>
#include <array>

#include "nn7feedforward.h"

#define CATCH_CONFIG_MAIN
#include <catch.hpp>

using std::cout;

#define NETWORK_SETTING_FILE "build/data/xor_feed_forward.xml"
#define NETWORK_TRAIN_FILE "build/data/xor_train.data"

#define MAX_EPOCH 500000
#define DESIRE_ERROR 0.0001
#define TRAIN_MOMENT 0.8
#define INPUTS_NUM 2
#define OUTPUT_NUM 1
#define HIDDEN_LAYOUT_NUM 1
#define NEURONS_NUM 3

void print_vector(NN7DataVector &n)
{
  for (int i = 0; i < n.size(); i++)
    cout << n[i] << " ";

  cout << "\n";
}

TEST_CASE("Base Neural network test") {

  SECTION("Train network XOR function") {
    NN7FeedForward<NN7UnipolarSigmoidNeuron> nn7(INPUTS_NUM, OUTPUT_NUM,
      HIDDEN_LAYOUT_NUM, NEURONS_NUM);

    nn7.setTrainMoment(TRAIN_MOMENT);

    for (;;) {
      try
      {
        nn7.trainNetworkOnFile(NETWORK_TRAIN_FILE, MAX_EPOCH, DESIRE_ERROR,
          NN7FeedForward<NN7UnipolarSigmoidNeuron>::RESILIENT_PROPAGATION_TRAINING);
      }
      catch (NN7Exception &e)
      {
        NN7Exception::printLastException(e);
      }

      cout << "Train epoch: " << nn7.getEpochNum() << "\n";
      if (nn7.getEpochNum() < MAX_EPOCH) {
        nn7.saveNetwork(NETWORK_SETTING_FILE);
        break;
      }

      nn7.randomizeWeight();
    }
  }

  SECTION("XOR network test") {

    NN7DataVector x1 { 0, 1 };
    NN7DataVector x2 { 1, 1 };
    NN7DataVector x3 { 0, 0 };
    NN7DataVector x4 { 1, 0 };
    NN7DataVector x5 { 0.5, 0.5 };
    NN7DataVector testOut;

    NN7FeedForward<NN7UnipolarSigmoidNeuron> newNetwork(NETWORK_SETTING_FILE);

    testOut = newNetwork.response(x1);
    cout << "(0, 1) -> " << testOut[0] << "\n";
    REQUIRE(testOut[0] > 0.9);

    testOut = newNetwork.response(x2);
    cout << "(1, 1) -> " << testOut[0] << "\n";
    REQUIRE(testOut[0] < 0.1);

    testOut = newNetwork.response(x3);
    cout << "(0, 0) -> " << testOut[0] << "\n";
    REQUIRE(testOut[0] < 0.1);

    testOut = newNetwork.response(x4);
    cout << "(1, 0) -> " << testOut[0] << "\n";
    REQUIRE(testOut[0] > 0.9);

    testOut = newNetwork.response(x5);
    cout << "(0.5, 0.5) -> " << testOut[0] << "\n";
  }

  SECTION("Test error processing") {
    NN7FeedForward<NN7UnipolarSigmoidNeuron> network(2, 1, 2, 3);
    try
    {
      NN7DataVector input { 1, 23, 45, 2};
      network.response(input);
    }
    catch (NN7Exception &e)
    {
      cout << "Incorrect input vector\n";
      NN7Exception::printAllException(e);
    }
  }
}
