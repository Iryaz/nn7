/*
Copyright (C) 2017 Ilya Ryasanov (ryasanov@gmail.com)
This program is free software; you can redistribute it and/or modify it
under the terms of the GNU General Public License as published by the
Free Software Foundation; either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for
more details.

You should have received a copy of the GNU General Public License along
with this program; if not, see <http://www.gnu.org/licenses>.
*/

#ifndef NN7_NN7FEEDFORWARD_H_
#define NN7_NN7FEEDFORWARD_H_

#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "nn7datavector.h"
#include "nn7layout.h"
#include "nn7sigmoidneuron.h"
#include "xmlconfig.h"

struct TrainData
{
  TrainData(NN7DataVector& in, NN7DataVector& out) :
    inputVector(in), requiredOutVector(out) {}

  NN7DataVector inputVector;
  NN7DataVector requiredOutVector;
};

typedef std::vector<TrainData> TrainDataList;

template <class NEURON_TYPE> class NN7FeedForward
{
public:

  enum TRAINING_ALGORITHM {
    BACK_PROPAGATION_TRAINING,
    RESILIENT_PROPAGATION_TRAINING
  };

  NN7FeedForward(int inputsNum, int outputsNum, int hiddenLayersNum, int neuronsNumHiddenLayer) :
    inputsNum_(inputsNum), outputsNum_(outputsNum), hiddenLayersNum_(hiddenLayersNum), neuronsNumHiddenLayer_(neuronsNumHiddenLayer)
  {
    initNeuralNetwork();
  }

  NN7FeedForward()
  {
    inputsNum_ = 0;
    outputsNum_ = 0;
    hiddenLayersNum_ = 0;
    neuronsNumHiddenLayer_ = 0;
  }

  NN7FeedForward(const char* xmlFile)
  {
    loadNetwork(xmlFile);
  }

  ~NN7FeedForward()
  {
    NN7Layout<NEURON_TYPE>* l = firstLayout_;
    int i = 0;
    while (l != NULL) {
      NN7Layout<NEURON_TYPE>* t = l;
      l = l->getNextLayer();
      delete t;
    }
  }

  void initNeuralNetwork()
  {
    randomGenerator_.setRange(-0.1, 0.1);
    trainEpochNum_ = 0;
    errorCost_ = 0;
    momentumConstant_ = 0.5f;
    trainMoment_ = 1.0f;

    firstLayout_ = new NN7Layout<NEURON_TYPE>(neuronsNumHiddenLayer_, inputsNum_);
    firstLayout_->setRandomizer(&randomGenerator_);

    NN7Layout<NEURON_TYPE>* l = firstLayout_;

    for (int i = 1; i < hiddenLayersNum_; i++) {
      l->setNextLayer(new NN7Layout<NEURON_TYPE>(neuronsNumHiddenLayer_, neuronsNumHiddenLayer_));
      l->getNextLayer()->setPrevLayer(l);
      l->getNextLayer()->setRandomizer(&randomGenerator_);
      l = l->getNextLayer();
    }

    l->setNextLayer(new NN7Layout<NEURON_TYPE>(outputsNum_, neuronsNumHiddenLayer_));
    l->getNextLayer()->setPrevLayer(l);
    l->getNextLayer()->setRandomizer(&randomGenerator_);

    lastLayout_ = l->getNextLayer();
  }

  void randomizeWeight()
  {
    randomGenerator_.setRange(-0.1, 0.1);
    NN7Layout<NEURON_TYPE>* l = firstLayout_;
    while (l != NULL) {
      l->setRandomizer(&randomGenerator_);
      l = l->getNextLayer();
    }
  }

  NN7DataVector response(NN7DataVector& in)
  {
    NN7Layout<NEURON_TYPE>* l = firstLayout_;
    try
    {
      l->setLayoutData(&in);
    }
    catch (NN7Exception& e)
    {
      RELAY_EXCEPTION(e, "NN7FeedForward", "response", "");
    }

    while (l != NULL) {
      l = l->nextResponse();
    }

    l = firstLayout_;

    while (l->getNextLayer() != NULL)
      l = l->getNextLayer();

    return l->getLastResponse();
  }

  int getInputsNum() const { return inputsNum_; }
  int getOutputNum() const { return outputsNum_; }
  int getHiddenLayersNum() const { return hiddenLayersNum_; }
  int getNeuronsNumHiddenLayer() const { return neuronsNumHiddenLayer_; }

  void setHiddenLayersNum(int hiddenLayersNum) {
    hiddenLayersNum_ = hiddenLayersNum;
  }
  void setNeuronsNumHiddenLayer(int neuronsNumHiddenLayer) {
    neuronsNumHiddenLayer_ = neuronsNumHiddenLayer;
  }
  void setInputsNum(int inputsNum) { inputsNum_ = inputsNum; }
  void setOutputsNum(int outputsNum) { outputsNum_ = outputsNum; }

  int getLayerNeuronsNum(int layerIndex) const
  {
    if (layerIndex == getHiddenLayersNum()) return getOutputNum();
    if (layerIndex > (getHiddenLayersNum() - 1)) return 0;
    int count = 0;
    NN7Layout<NEURON_TYPE>* l = firstLayout_;

    while (l != NULL && count != layerIndex) {
      l = l->getNextLayer();
      count++;
    }

    return l->getNeuronsNum();
  }

  double getWeight(int hiddenLayerIndex, int neuronIndex, int neuronInputIndex)
  {
    if (hiddenLayerIndex > getHiddenLayersNum()) return 0;
    int lcount = 0;
    NN7Layout<NEURON_TYPE>* l = firstLayout_;

    while (l != NULL && lcount != hiddenLayerIndex) {
      l = l->getNextLayer();
      lcount++;
    }

    if (neuronIndex > (l->getNeuronsNum() - 1)) return 0;
    NEURON_TYPE* neuron = l->getNeuron(neuronIndex);
    if (neuronInputIndex > (neuron->getInputsNum())) return 0;
    return neuron->getWeight(neuronInputIndex);
  }

  void setWeight(int layersIndex, int neuronIndex, int neuronInputIndex, double newWeight)
  {
    if (layersIndex > getHiddenLayersNum()) return;
    int lcount = 0;
    NN7Layout<NEURON_TYPE>* l = firstLayout_;

    while (l != NULL && lcount != layersIndex) {
      l = l->getNextLayer();
      lcount++;
    }

    if (neuronIndex > (l->getNeuronsNum() - 1)) return;
    NEURON_TYPE* neuron = l->getNeuron(neuronIndex);
    if (neuronInputIndex > (neuron->getInputsNum())) return;
    neuron->setWeight(newWeight, neuronInputIndex);
  }

  void setMomentumConst(double _moment) { momentumConstant_ = _moment; }
  double getMomentumConst() const { return momentumConstant_; }

  void setTrainMoment(double newMoment) { trainMoment_ = newMoment; }
  double getTrainMoment() const { return trainMoment_; }

  double getErrorCost() const { return errorCost_ / outputsNum_; }

  void trainNetwork(TrainDataList &data, int maxEpoch, double desiredError, TRAINING_ALGORITHM alg)
  {
    trainEpochNum_ = 0;

    if (maxEpoch < 0)
      THROW_EXCEPTION("NN7FeedForward", "trainNetwork", "maxEpoch < 0");
    if (desiredError <= 0)
      THROW_EXCEPTION("NN7FeedForward", "trainNetwork", "desiredError <= 0");

    for (int e = 0; e < maxEpoch; e++) {

      errorCost_ = 0;

      for (int d = 0; d < data.size(); d++) {
        trainOneEpoch(data[d]);
        updateDeltaWeights(data[d].inputVector);
        if (alg == BACK_PROPAGATION_TRAINING)
          correctWeights_backprop();
      }

      if (alg == RESILIENT_PROPAGATION_TRAINING)
        correctWeights_rprop();

      trainEpochNum_++;

      if (getErrorCost() <= desiredError) break;
    }
  }

  void trainNetworkOnFile(const char* fileName, int maxEpoch, double desiredError, TRAINING_ALGORITHM alg)
  {
    TrainDataList trainData;
    std::string line;
    std::fstream trainFile(fileName, std::fstream::in);
    int trainDataSetCount = 0;

    int dataSetSize = 0, inputNum = 0, outputsNum_ = 0;
    int d = 0;
    if (trainFile.is_open())
    {
      std::string oneString;
      std::getline(trainFile, oneString);
      std::stringstream s(oneString);

      s >> dataSetSize;
      s >> inputNum;
      s >> outputsNum_;

      if (dataSetSize <= 0)
        THROW_EXCEPTION("NN7FeedForward", "trainNetworkOnFile",
          "train data set == 0 (" + oneString + ")");

      if (inputNum != NN7FeedForward::inputsNum_)
        THROW_EXCEPTION("NN7FeedForward", "trainNetworkOnFile",
          "input num != network input num");

      if (outputsNum_ != NN7FeedForward::outputsNum_)
        THROW_EXCEPTION("NN7FeedForward", "trainNetwork",
          "out num != network out num");

      while (std::getline(trainFile, oneString) && trainDataSetCount < dataSetSize) {

        NN7DataVector xVector;
        NN7DataVector yVector;
        std::stringstream s1(oneString);

        for (int i = 0; i < inputNum; i++) {
          double x = 0;
          s1 >> x;
          xVector.append(x);
        }

        if (!std::getline(trainFile, oneString))
          THROW_EXCEPTION("NN7FeedForward", "trainNetworkOnFile",
            "Uncorrect train file (train data set > file size)");

        std::stringstream s2(oneString);
        for (int i = 0; i < outputsNum_; i++) {
          double y = 0;
          s2 >> y;
          yVector.append(y);
        }

        trainData.push_back(TrainData(xVector, yVector));
        trainDataSetCount++;
      }

      trainFile.close();

      try
      {
        trainNetwork(trainData, maxEpoch, desiredError, alg);
      }
      catch (NN7Exception& e)
      {
        RELAY_EXCEPTION(e, "NN7FeedForward", "trainNetworkOnFile", "Error train network");
      }
    }
    else
      THROW_EXCEPTION("NN7FeedForward", "trainNetworkOnFile", "Error open file - " +
        std::string(fileName));
  }

  int getEpochNum() const { return trainEpochNum_; }

  void updateDeltaWeights(NN7DataVector& trainData)
  {
    NN7Layout<NEURON_TYPE>* l = firstLayout_;

    while (l != NULL) {
      for (int n = 0; n < l->getNeuronsNum(); n++) {
        NEURON_TYPE *currentNeuron = l->getNeuron(n);

        for (int w = 0; w < currentNeuron->getInputsNum(); w++) {
          double E = 0;
          double weight = currentNeuron->getWeight(w);
          if (l == firstLayout_)
            E = trainData[w] * l->getGradients()->operator[](n) * -1;
          else
            E = l->getPrevLayer()->getNeuron(w)->getLastResponse() *
              l->getGradients()->operator[](n) * -1;

          double dE = l->getDerivateE(n, w);
          l->setDerivateE(dE + E, n, w);
        }
      }

      l = l->getNextLayer();
    }
  }

  double sign(double x)
  {
    if (x == 0.0) return 0.0;
    if (x > 0.0) return 1.0;
    if (x < 0.0) return -1.0;
  }

  void correctWeights_backprop()
  {
      NN7Layout<NEURON_TYPE>* l = firstLayout_;

      while (l != NULL) {
        for (int n = 0; n < l->getNeuronsNum(); n++) {
          NEURON_TYPE* currentNeuron = l->getNeuron(n);
          for (int w = 0; w < currentNeuron->getInputsNum(); w++) {
            double deltaWeight = l->getDerivateE(n, w);
            double oldWeight = currentNeuron->getWeight(w);
            double prevDeltaW = l->getPrevDeltaWeights(n, w);

            deltaWeight *= trainMoment_;
            deltaWeight += momentumConstant_ * prevDeltaW;
            currentNeuron->setWeight(oldWeight + deltaWeight, w);
            l->setPrevDeltaWeight(deltaWeight, n, w);
            l->setDerivateE(0, n, w);
          }
        }
        l = l->getNextLayer();
      }
  }

  void correctWeights_rprop()
  {
    NN7Layout<NEURON_TYPE>* l = firstLayout_;

    while (l != NULL) {
      for (int n = 0; n < l->getNeuronsNum(); n++) {
        NEURON_TYPE *currentNeuron = l->getNeuron(n);
        for (int w = 0; w < currentNeuron->getInputsNum(); w++) {
          double dE = l->getDerivateE(n, w);
          double prev_dE = l->getPrevDerivateE(n, w);
          double delta = l->getTrainRPROPMoment(n, w);

          if (dE * prev_dE > 0) {
            delta = std::min<double>(delta * 1.2, 50);
          }

          if (dE * prev_dE < 0) {
            delta = std::max<double>(delta * 0.5, 0.000001);
          }

          double deltaW = 0;
          if (dE > 0) {
            deltaW = delta;
          }
          if( dE < 0) {
            deltaW = -delta;
          }
          if (dE == 0) {
            deltaW = 0;
          }

          double oldW = currentNeuron->getWeight(w);
          currentNeuron->setWeight(oldW + deltaW, w);
          l->setPrevDerivateE(dE, n, w);
          l->setDerivateE(0, n, w);
          l->setTrainRPROPMoment(delta, n, w);
        }
      }

      l = l->getNextLayer();
    }
  }

  void saveNetwork(const char* xmlFile)
  {
    network2XML<NN7FeedForward<NEURON_TYPE>>(*this, xmlFile);
  }

  void loadNetwork(const char* xmlFile)
  {
    XML2Network<NN7FeedForward<NEURON_TYPE>>(*this, xmlFile);
  }

  template <class N> friend void print_nn(NN7FeedForward<N>& nn);

protected:
  NN7Layout<NEURON_TYPE>* firstLayout_;
  NN7Layout<NEURON_TYPE>* lastLayout_;

  void trainOneEpoch(TrainData& oneTrain)
  {
    try
    {
      NN7Layout<NEURON_TYPE>* l = lastLayout_;
      NN7DataVector out = response(oneTrain.inputVector);
      NN7DataVector errorVector = out - oneTrain.requiredOutVector;

      for (int n = 0; n < l->getNeuronsNum(); n++) {
        std::vector<double>* gradients = l->getGradients();
        NEURON_TYPE *currentNeuron = l->getNeuron(n);
        gradients->operator[](n) = errorVector[n] *
          currentNeuron->getDerivative();

        errorCost_ += pow(errorVector[n], 2);
      }

      l = l->getPrevLayer();

      while (l != NULL) {

        for (int i = 0; i < l->getNeuronsNum(); i++) {

          double gradient = 0;
          NN7Layout<NEURON_TYPE>* nextLayout = l->getNextLayer();
          NEURON_TYPE* currentNeuron = l->getNeuron(i);

          for (int j = 0; j < nextLayout->getNeuronsNum(); j++) {
            gradient += nextLayout->getNeuron(j)->getWeight(i) *
              nextLayout->getGradients()->operator[](j);
          }

          l->getGradients()->operator[](i) = gradient *
            currentNeuron->getDerivative();
        }

        l = l->getPrevLayer();
      }
    }
    catch (NN7Exception& e)
    {
      RELAY_EXCEPTION(e, "NN7FeedForward", "trainOneEpoch", "Exception");
    }
  }

private:
  int hiddenLayersNum_;
  int inputsNum_;
  int outputsNum_;
  int neuronsNumHiddenLayer_;
  NN7Randomizer randomGenerator_;

  double momentumConstant_;
  double errorCost_;
  double trainMoment_;
  int trainEpochNum_;

};

template <class NEURON_TYPE> void print_nn(NN7FeedForward<NEURON_TYPE>& nn)
{
  if (nn.hiddenLayersNum_ == 0) return;
  NN7Layout<NEURON_TYPE>* l = nn.firstLayout_;

  std::cout << "\nNeuron network -------------> \n";
  for ( int i = 0; i <= nn.hiddenLayersNum_; i++) {
    std::cout << "Layout N" << i << "\n";
    for (int j = 0; j < l->getNeuronsNum(); j++) {
      std::cout << " Neuron N" << j << " Gradient: ";
      std::cout << l->getGradients()->operator[](j) << " ";

      NN7LinearNeuron::print_weight(*(l->getNeuron(j)));
    }

    l = l->getNextLayer();
  }
}

#endif
