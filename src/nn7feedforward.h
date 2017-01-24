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

  NN7FeedForward(uint32_t inputsNum, uint32_t outputsNum, uint32_t hiddenLayersNum, uint32_t neuronsNumHiddenLayer) :
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
    uint32_t i = 0;
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

    for (uint32_t i = 1; i < hiddenLayersNum_; i++) {
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

    return lastLayout_->getLastResponse();
  }

  uint32_t getInputsNum() const { return inputsNum_; }
  uint32_t getOutputNum() const { return outputsNum_; }
  uint32_t getHiddenLayersNum() const { return hiddenLayersNum_; }
  uint32_t getNeuronsNumHiddenLayer() const { return neuronsNumHiddenLayer_; }

  void setHiddenLayersNum(uint32_t hiddenLayersNum) {
    hiddenLayersNum_ = hiddenLayersNum;
  }
  void setNeuronsNumHiddenLayer(uint32_t neuronsNumHiddenLayer) {
    neuronsNumHiddenLayer_ = neuronsNumHiddenLayer;
  }
  void setInputsNum(uint32_t inputsNum) { inputsNum_ = inputsNum; }
  void setOutputsNum(uint32_t outputsNum) { outputsNum_ = outputsNum; }

  uint32_t getLayerNeuronsNum(uint32_t layerIndex) const
  {
    if (layerIndex == getHiddenLayersNum()) return getOutputNum();
    if (layerIndex > (getHiddenLayersNum() - 1)) return 0;
    uint32_t count = 0;
    NN7Layout<NEURON_TYPE>* l = firstLayout_;

    while (l != NULL && count != layerIndex) {
      l = l->getNextLayer();
      count++;
    }

    return l->getNeuronsNum();
  }

  double getWeight(uint32_t hiddenLayerIndex, uint32_t neuronIndex, uint32_t neuronInputIndex)
  {
    if (hiddenLayerIndex > getHiddenLayersNum()) return 0;
    uint32_t lcount = 0;
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

  void setWeight(uint32_t layersIndex, uint32_t neuronIndex, uint32_t neuronInputIndex, double newWeight)
  {
    if (layersIndex > getHiddenLayersNum()) return;
    uint32_t lcount = 0;
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

  double getBias(uint32_t hiddenLayerIndex, uint32_t neuronIndex)
  {
    if (hiddenLayerIndex > getHiddenLayersNum()) return 0;
    uint32_t lcount = 0;
    NN7Layout<NEURON_TYPE>* l = firstLayout_;

    while (l != NULL && lcount != hiddenLayerIndex) {
      l = l->getNextLayer();
      lcount++;
    }

    if (neuronIndex > (l->getNeuronsNum() - 1)) return 0;

    NEURON_TYPE* neuron = l->getNeuron(neuronIndex);
    return neuron->getBias();
  }

  void setBias(uint32_t layersIndex, uint32_t neuronIndex, double newBias)
  {
    if (layersIndex > getHiddenLayersNum()) return;
    uint32_t lcount = 0;
    NN7Layout<NEURON_TYPE>* l = firstLayout_;

    while (l != NULL && lcount != layersIndex) {
      l = l->getNextLayer();
      lcount++;
    }

    if (neuronIndex > (l->getNeuronsNum() - 1)) return;
    NEURON_TYPE* neuron = l->getNeuron(neuronIndex);
    neuron->setBias(newBias);
  }

  void setMomentumConst(double _moment) { momentumConstant_ = _moment; }
  double getMomentumConst() const { return momentumConstant_; }

  void setTrainMoment(double newMoment) { trainMoment_ = newMoment; }
  double getTrainMoment() const { return trainMoment_; }

  double getErrorCost() const { return errorCost_;}

  void trainNetwork(TrainDataList &data, uint32_t maxEpoch, double desiredError)
  {
    trainEpochNum_ = 0;

    if (maxEpoch < 0)
      THROW_EXCEPTION("NN7FeedForward", "trainNetwork", "maxEpoch < 0");
    if (desiredError <= 0)
      THROW_EXCEPTION("NN7FeedForward", "trainNetwork", "desiredError <= 0");

    for (uint32_t e = 0; e < maxEpoch; e++) {
      errorCost_ = 0;
      for (uint32_t d = 0; d < data.size(); d++) {
        trainOneEpoch(data[d]);
      }

      trainEpochNum_++;
	  errorCost_ /= data.size();

      if (getErrorCost() <= desiredError) break;
    }
  }

  void trainNetworkOnFile(const char* fileName, uint32_t maxEpoch, double desiredError)
  {
    TrainDataList trainData;
    std::string line;
    std::fstream trainFile(fileName, std::fstream::in);
    uint32_t trainDataSetCount = 0;

    uint32_t dataSetSize = 0, inputNum = 0, outputsNum_ = 0;
    uint32_t d = 0;
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

        for (uint32_t i = 0; i < inputNum; i++) {
          double x = 0;
          s1 >> x;
          xVector.append(x);
        }

        if (!std::getline(trainFile, oneString))
          THROW_EXCEPTION("NN7FeedForward", "trainNetworkOnFile",
            "Uncorrect train file (train data set > file size)");

        std::stringstream s2(oneString);
        for (uint32_t i = 0; i < outputsNum_; i++) {
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
        trainNetwork(trainData, maxEpoch, desiredError);
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

  uint32_t getEpochNum() const { return trainEpochNum_; }

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

  void updateWeightBackProp(NN7Layout<NEURON_TYPE>* layer, uint32_t neuron, uint32_t weight, double x)
  {
	  NEURON_TYPE* currentNeuron = layer->getNeuron(neuron);
	  double oldW = currentNeuron->getWeight(weight);
	  double deltaW = x * layer->getGradient(neuron) * trainMoment_;
	  deltaW += momentumConstant_ * currentNeuron->getPrevDeltaWeight(weight);
	  currentNeuron->setWeight(oldW + deltaW, weight);
	  currentNeuron->setPrevDeltaWeight(deltaW, weight);
  }

  void trainOneEpoch(TrainData& oneTrain)
  {
    try
    {
      NN7Layout<NEURON_TYPE>* l = lastLayout_;
      NN7DataVector out = response(oneTrain.inputVector);
      NN7DataVector errorVector = oneTrain.requiredOutVector - out;

      for (uint32_t n = 0; n < l->getNeuronsNum(); n++) {
        NEURON_TYPE *lastLayerNeuron = l->getNeuron(n);
        l->setGradient(errorVector[n] * lastLayerNeuron->getDerivative(), n);
		//double oldBias = lastLayerNeuron->getBias();
		//lastLayerNeuron->setBias(oldBias + (l->getGradient(n) * trainMoment_));

		NN7Layout<NEURON_TYPE> *p = l->getPrevLayer();
		for (uint32_t w = 0; w < lastLayerNeuron->getInputsNum(); w++)
			updateWeightBackProp(l, n, w, p->getNeuron(w)->getLastResponse());

        errorCost_ += pow(errorVector[n], 2);
      }

      l = l->getPrevLayer();

      while (l != NULL) {
        for (uint32_t i = 0; i < l->getNeuronsNum(); i++) {

          double gradient = 0;
          NN7Layout<NEURON_TYPE>* nextLayout = l->getNextLayer();
          NEURON_TYPE* currentNeuron = l->getNeuron(i);

          for (uint32_t j = 0; j < nextLayout->getNeuronsNum(); j++) {
            gradient += nextLayout->getNeuron(j)->getWeight(i) *
              nextLayout->getGradient(j);
          }

          l->setGradient(gradient *currentNeuron->getDerivative(), i);
		  //double oldBias = currentNeuron->getBias();
		  //currentNeuron->setBias(oldBias + (l->getGradient(i) * trainMoment_));
		  
		  if (l->getPrevLayer() == NULL) {
			for (uint32_t w = 0; w < currentNeuron->getInputsNum(); w++)
				updateWeightBackProp(l, i, w, oneTrain.inputVector[w]);
		  } 
		  else
		  {
			NN7Layout<NEURON_TYPE>* p = l->getPrevLayer();  
			for (uint32_t w = 0; w < currentNeuron->getInputsNum(); w++)
				updateWeightBackProp(l, i, w, p->getNeuron(w)->getLastResponse());
		  }
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
  uint32_t hiddenLayersNum_;
  uint32_t inputsNum_;
  uint32_t outputsNum_;
  uint32_t neuronsNumHiddenLayer_;
  NN7Randomizer randomGenerator_;

  double momentumConstant_;
  double errorCost_;
  double trainMoment_;
  uint32_t trainEpochNum_;

};

template <class NEURON_TYPE> void print_nn(NN7FeedForward<NEURON_TYPE>& nn)
{
  if (nn.hiddenLayersNum_ == 0) return;
  NN7Layout<NEURON_TYPE>* l = nn.firstLayout_;

  std::cout << "\nNeuron network -------------> \n";
  for ( uint32_t i = 0; i <= nn.hiddenLayersNum_; i++) {
    std::cout << "Layout N" << i << "\n";
    for (uint32_t j = 0; j < l->getNeuronsNum(); j++) {
      std::cout << " Neuron N" << j << " Bias: ";
      std::cout << l->getNeuron(j)->getBias() << " ";

      NN7LinearNeuron::print_weight(*(l->getNeuron(j)));
    }

    l = l->getNextLayer();
  }
}

#endif
