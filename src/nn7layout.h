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

#ifndef NN7_NN7LAYOUT_H_
#define NN7_NN7LAYOUT_H_

#include <vector>

#include "nn7datavector.h"
#include "nn7randomizer.h"

template <class NEURON_TYPE> class NN7Layout
{
public:

  NN7Layout(int neuronsNum, int neuronInputNum)
  {
    localGradients_ = new std::vector<double>(neuronsNum);
    prevDeltaWeightLst_ = new std::vector<double>(neuronsNum * neuronInputNum, 0.0);
    derivateELst_ = new std::vector<double>(neuronsNum * neuronInputNum, 0.0);
    prevDerivateELst_ = new std::vector<double>(neuronInputNum * neuronInputNum, 0.0);
    trainRPROPSteps_ = new std::vector<double>(neuronInputNum * neuronInputNum, 0.0);

    nextLayout_ = NULL;
    prevLayout_ = NULL;
    neuronInputNum_ = neuronInputNum;
    neuronsNum_ = neuronsNum;

    for (int i = 0; i < neuronsNum_; i++) {
      NEURON_TYPE* n = new NEURON_TYPE(neuronInputNum_);
      neuronLst_.push_back(n);
    }
  }

  ~NN7Layout()
  {
    for (int i = 0; i < neuronLst_.size(); i++)
      delete neuronLst_[i];

    delete localGradients_;
    delete prevDeltaWeightLst_;
    delete derivateELst_;
    delete prevDerivateELst_;
    delete trainRPROPSteps_;
  }

  NN7Layout<NEURON_TYPE>* nextResponse()
  {
    if (nextLayout_ != NULL) {
      NN7DataVector d = getLastResponse();
      nextLayout_->setLayoutData(&d);
    }
    return nextLayout_;
  }

  void setRandomizer(NN7Randomizer* randomGen)
  {
    for (int i = 0; i < neuronLst_.size(); i++)
    {
      neuronLst_[i]->setRandomizer(randomGen);
    }
  }

  void setLayoutData(NN7DataVector* data)
  {
    try
    {
      for (int i = 0; i < neuronLst_.size(); i++)
        neuronLst_[i]->response(data);
    }
    catch (NN7Exception& e)
    {
      RELAY_EXCEPTION(e, "NN7Layout", "setLayoutData", "");
    }
  }

  NN7DataVector& getLayoutResponse()
  {
    NN7DataVector ret;
    for (int i = 0; i < neuronLst_.size(); i++)
      ret.append(neuronLst_[i]->getLastResponse());

    return ret;
  }

  NEURON_TYPE* operator[](int neuronIndex) const
  {
    if (neuronIndex > (neuronLst_.size() -1))
      THROW_EXCEPTION("NN7Layout", "getNeuron", "layout::size() < index");

    return neuronLst_[neuronIndex];
  }

  NEURON_TYPE* getNeuron(int neuronIndex) const
  {
    if (neuronIndex > (neuronLst_.size() -1))
      THROW_EXCEPTION("NN7Layout", "getNeuron", "layout::size() < index");

    return neuronLst_[neuronIndex];
  }

  NN7DataVector getLastResponse() const
  {
    NN7DataVector ret;
    for (int i = 0; i < neuronLst_.size(); i++)
      ret.append(neuronLst_[i]->getLastResponse());

    return ret;
  }

  NN7Layout<NEURON_TYPE>* getNextLayer() const { return nextLayout_; }
  NN7Layout<NEURON_TYPE>* getPrevLayer() const { return prevLayout_; }
  std::vector<double>* getGradients() const { return localGradients_; }

  double getPrevDeltaWeights(int neuronIndex, int weightIndex) const
  {
    if (neuronIndex > (neuronLst_.size() -1))
      THROW_EXCEPTION("NN7Layout", "getPrevDeltaWeights", "Neuron index - incorrect");

    if (weightIndex > (getNeuron(neuronIndex)->getInputsNum() -1))
      THROW_EXCEPTION("NN7Layout", "setPrevDeltaWeight", "Weight index - incorrect");

    return prevDeltaWeightLst_->operator[](weightIndex*(neuronIndex+1));
  }

  void setPrevDeltaWeight(double prevWeight, int neuronIndex, int weightIndex)
  {
    if (neuronIndex > (neuronLst_.size() -1))
      THROW_EXCEPTION("NN7Layout", "setPrevDeltaWeight", "Neuron index - incorrect");

    if (weightIndex > (getNeuron(neuronIndex)->getInputsNum() -1))
      THROW_EXCEPTION("NN7Layout", "setPrevDeltaWeight", "Weight index - incorrect");

    prevDeltaWeightLst_->operator[](weightIndex*(neuronIndex+1)) = prevWeight;
  }

  double getDerivateE(int neuronIndex, int weightIndex) const
  {
    if (neuronIndex > (neuronLst_.size() -1))
      THROW_EXCEPTION("NN7Layout", "getDerivateE", "Neuron index - incorrect");

    if (weightIndex > (getNeuron(neuronIndex)->getInputsNum() -1))
      THROW_EXCEPTION("NN7Layout", "getDerivateE", "Weight index - incorrect");

    return derivateELst_->operator[](weightIndex*(neuronIndex+1));
  }

  void setDerivateE(double newValue, int neuronIndex, int weightIndex)
  {
    if (neuronIndex > (neuronLst_.size() -1))
      THROW_EXCEPTION("NN7Layout", "setDerivateE", "Neuron index - incorrect");

    if (weightIndex > (getNeuron(neuronIndex)->getInputsNum() -1))
      THROW_EXCEPTION("NN7Layout", "setDerivateE", "Weight index - incorrect");

    derivateELst_->operator[](weightIndex*(neuronIndex+1)) = newValue;
  }

  double getPrevDerivateE(int neuronIndex, int weightIndex) const
  {
    if (neuronIndex > (neuronLst_.size() -1))
      THROW_EXCEPTION("NN7Layout", "getPrevDerivateE", "Neuron index - incorrect");

    if (weightIndex > (getNeuron(neuronIndex)->getInputsNum() -1))
      THROW_EXCEPTION("NN7Layout", "getPrevDerivateE", "Weight index - incorrect");

    return prevDerivateELst_->operator[](weightIndex*(neuronIndex+1));
  }

  void setPrevDerivateE(double newValue, int neuronIndex, int weightIndex)
  {
    if (neuronIndex > (neuronLst_.size() -1))
      THROW_EXCEPTION("NN7Layout", "setPrevDerivateE", "Neuron index - incorrect");

    if (weightIndex > (getNeuron(neuronIndex)->getInputsNum() -1))
      THROW_EXCEPTION("NN7Layout", "setPrevDerivateE", "Weight index - incorrect");

    prevDerivateELst_->operator[](weightIndex*(neuronIndex+1)) = newValue;
  }

  double getTrainRPROPMoment(int neuronIndex, int weightIndex) const
  {
    if (neuronIndex > (neuronLst_.size() -1))
      THROW_EXCEPTION("NN7Layout", "getTrainRPROPMoment", "Neuron index - incorrect");

    if (weightIndex > (getNeuron(neuronIndex)->getInputsNum() -1))
      THROW_EXCEPTION("NN7Layout", "getTrainRPROPMoment", "Weight index - incorrect");

    return trainRPROPSteps_->operator[](weightIndex*(neuronIndex+1));
  }

  void setTrainRPROPMoment(double newValue, int neuronIndex, int weightIndex)
  {
    if (neuronIndex > (neuronLst_.size() -1))
      THROW_EXCEPTION("NN7Layout", "setTrainRPROPMoment", "Neuron index - incorrect");

    if (weightIndex > (getNeuron(neuronIndex)->getInputsNum() -1))
      THROW_EXCEPTION("NN7Layout", "setTrainRPROPMoment", "Weight index - incorrect");

    trainRPROPSteps_->operator[](weightIndex*(neuronIndex+1)) = newValue;
  }

  int getNeuronsNum() const { return neuronsNum_; }
  int getInputsNum() const { return neuronInputNum_; }

  void setNextLayer(NN7Layout<NEURON_TYPE>* nextLayer) { nextLayout_ = nextLayer; }
  void setPrevLayer(NN7Layout<NEURON_TYPE>* prevLayer) { prevLayout_ = prevLayer; }

private:
  std::vector<NEURON_TYPE*> neuronLst_;
  std::vector<double>* localGradients_;

  std::vector<double>* prevDeltaWeightLst_;
  std::vector<double>* derivateELst_;
  std::vector<double>* prevDerivateELst_;
  std::vector<double>* trainRPROPSteps_; // Only use rprop training

  int neuronInputNum_;
  int neuronsNum_;

  NN7Layout<NEURON_TYPE>* nextLayout_;
  NN7Layout<NEURON_TYPE>* prevLayout_;
};

#endif
