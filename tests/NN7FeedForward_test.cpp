
#include <iostream>
#include <array>

#include "nn7feedforward.h"

#define CATCH_CONFIG_MAIN
#include <catch.hpp>

using std::cout;

#define NETWORK_SETTING_FILE "xor_feed_forward.xml"
#define NETWORK_TRAIN_FILE "xor_train.data"

#define MAX_EPOCH 200000
#define DESIRE_ERROR 0.00001
#define TRAIN_MOMENT 0.4
#define MOMENTUM_CONST 0.0
#define INPUTS_NUM 2
#define OUTPUT_NUM 1
#define HIDDEN_LAYOUT_NUM 1
#define NEURONS_NUM 3

typedef NN7FeedForward<NN7UnipolarSigmoidNeuron> FeedForwardNetwork;

void print_vector(NN7DataVector &n)
{
  for (uint32_t i = 0; i < n.size(); i++)
    cout << n[i] << " ";

  cout << "\n";
}

TEST_CASE("Base Neural network test") {

  SECTION("Train network XOR function") {
	FeedForwardNetwork nn7(INPUTS_NUM, OUTPUT_NUM, HIDDEN_LAYOUT_NUM, NEURONS_NUM);

    nn7.setTrainMoment(TRAIN_MOMENT);
    nn7.setMomentumConst(MOMENTUM_CONST);
    print_nn(nn7);
    try
    {
      nn7.trainNetworkOnFile(NETWORK_TRAIN_FILE, MAX_EPOCH, DESIRE_ERROR);
    }
    catch (NN7Exception &e)
    {
      NN7Exception::printLastException(e);
    }
	print_nn(nn7);
    cout << "Train epoch: " << nn7.getEpochNum() << "\n";
	cout << "Error Cost: " << nn7.getErrorCost() << "\n";
    nn7.saveNetwork(NETWORK_SETTING_FILE);
  }

  SECTION("XOR network test") {

    NN7DataVector x1 { 1, 1 };
    NN7DataVector x2 { 0, 0 };
    NN7DataVector x3 { 1, 0 };
    NN7DataVector x4 { 0, 1 };
    NN7DataVector testOut;

	FeedForwardNetwork newNetwork(NETWORK_SETTING_FILE);

    testOut = newNetwork.response(x1);
    cout << "(1, 1) -> " << testOut[0] << "\n";
    REQUIRE(testOut[0] < 0.1);

    testOut = newNetwork.response(x2);
    cout << "(0, 0) -> " << testOut[0] << "\n";
    REQUIRE(testOut[0] < 0.1);

    testOut = newNetwork.response(x3);
    cout << "(1, 0) -> " << testOut[0] << "\n";
    REQUIRE(testOut[0] > 0.9);

    testOut = newNetwork.response(x4);
    cout << "(0, 1) -> " << testOut[0] << "\n";
    REQUIRE(testOut[0] > 0.9);
  }

  SECTION("Test error processing") {
    NN7FeedForward<NN7BipolarSigmoidNeuron> network(2, 1, 2, 3);
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
