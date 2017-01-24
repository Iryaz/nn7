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

#include "nn7linearneuron.h"

#include <iostream>

using std::cout;

NN7LinearNeuron::NN7LinearNeuron(uint32_t inputsNum, NN7Randomizer* randomGenerator)
    : inputsNum_(inputsNum), randomGenerator_(randomGenerator),
    lastInputDataPtr_(NULL), lastResponse_(0), lastV_(0), error_(0),
    bias_(0), prevBias_(0)
{
    weightLst_ = new double[inputsNum];
    prevDeltaWeight_ = new double[inputsNum_];

    if(randomGenerator_ != NULL)
      randomizeWeightLst();
    else {
      for(uint32_t i = 0; i < inputsNum_; i++) {
        weightLst_[i] = 0;
        prevDeltaWeight_[i] = 0;
      }
    }
}

NN7LinearNeuron::~NN7LinearNeuron()
{
  delete[] weightLst_;
  delete[] prevDeltaWeight_;
}

void NN7LinearNeuron::randomizeWeightLst()
{
  for (uint32_t i = 0; i < inputsNum_; i++)
    weightLst_[i] = randomGenerator_->getRandom();
}

void NN7LinearNeuron::randomWeight()
{
  randomizeWeightLst();
}

double NN7LinearNeuron::getDerivative()
{
  return 1;
}

double NN7LinearNeuron::getV(NN7DataVector* x)
{
    lastV_ = 0.0f;

    if (x->size() != inputsNum_) {
      THROW_EXCEPTION("NN7LinearNeuron", "getV",
        "x::size() != neuronInputsNum");
    }

    for (uint32_t i = 0; i < inputsNum_; i++)
      lastV_ += (x->getValue(i) * weightLst_[i]);

    return lastV_;
}

double NN7LinearNeuron::response(NN7DataVector* x)
{
  try
  {
    lastResponse_ = getV(x) + bias_;
  }
  catch (NN7Exception& e)
  {
    RELAY_EXCEPTION(e, "NN7LinearNeuron", "response",
      "getV(x): uncorrect x vector");
  }

  return lastResponse_;
}

void NN7LinearNeuron::setWeight(double newWeight, uint32_t weightIndex)
{
    if (weightIndex > (inputsNum_ - 1))
      THROW_EXCEPTION("NN7LinearNeuron", "setWeight",
        "weightIndex > neuronInputsNum");

    if (weightIndex < 0)
      THROW_EXCEPTION("NN7LinearNeuron", "setWeight",
        "weightIndex < 0");

    weightLst_[weightIndex] = newWeight;
}

void NN7LinearNeuron::setWeight(NN7DataVector* newWeightLst)
{
    if (newWeightLst->size()  != inputsNum_) {
      THROW_EXCEPTION("NN7LinearNeuron", "setWeight",
        "newWeightLst::size() != neuronInputsNum");
    }

    for (uint32_t i = 0; i < newWeightLst->size(); i++)
      weightLst_[i] = newWeightLst->getValue(i);
}

double NN7LinearNeuron::getWeight(uint32_t weightIndex) const
{
  if (weightIndex > (inputsNum_ - 1) || weightIndex < 0) {

    if (weightIndex > (inputsNum_ - 1))
      THROW_EXCEPTION("NN7LinearNeuron", "getWeight",
        "weightIndex > neuronInputsNum");

    if (weightIndex < 0)
    THROW_EXCEPTION("NN7LinearNeuron", "getWeight",
      "weightIndex < 0");
  }

  return weightLst_[weightIndex];
}

double NN7LinearNeuron::operator[](uint32_t weightIndex) const
{
  return getWeight(weightIndex);
}

void NN7LinearNeuron::setRandomizer(NN7Randomizer* randomGenerator)
{
  randomGenerator_ = randomGenerator;

  if(randomGenerator_ != NULL)
    randomizeWeightLst();
}

void NN7LinearNeuron::print_weight(NN7LinearNeuron& n)
{
  cout << "Weight neuron: ";
  for (uint32_t i = 0; i < n.inputsNum_; i++) {
    cout << n[i] << " ";
  }

  cout << "\n";
}
