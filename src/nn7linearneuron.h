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

#ifndef NN7_NN7LINEAR_NEURON_H_
#define NN7_NN7LINEAR_NEURON_H_

#include <random>

#include "nn7datavector.h"
#include "nn7exception.h"
#include "nn7randomizer.h"

#define DEFAULT_WEIGHT_RANGE 1.0

class NN7LinearNeuron
{
public:
  NN7LinearNeuron(int inputsNum, NN7Randomizer* randomizer = NULL);
  ~NN7LinearNeuron();

  void setWeight(double newWeight, int weightIndex);
  void setWeight(NN7DataVector* newWeightLst);
  void setError(double error) { error_ = error; }
  void setRandomizer(NN7Randomizer* randomizer);
  void randomWeight();
  void setBias(double bias) { bias_ = bias; }
  double getBias() const { return bias_; }
  void setPrevBias(double bias) { prevBias_ = bias; }
  double getPrevBias() const { return prevBias_; }
  double getV(NN7DataVector* x);
  virtual double getDerivative();
  virtual double response(NN7DataVector* x);
  double getLastResponse() const { return lastResponse_; }
  double getLastV() const { return lastV_; }
  double getWeight(int weightIndex) const;
  double operator[](int weightLst) const;
  double getError() { return error_; }
  double getPrevDeltaWeight(int inputIndex) const { return prevDeltaWeight_[inputIndex]; }
  void setPrevDeltaWeight(double weight, int inputIndex) { prevDeltaWeight_[inputIndex] = weight; }

  int getInputsNum() const  { return inputsNum_; }

  static void print_weight(NN7LinearNeuron& n);

  friend void print_weight(NN7LinearNeuron& n);

protected:
  void randomizeWeightLst();

  double lastResponse_;
  double lastV_;
  double error_;
  double* weightLst_;
  double* prevDeltaWeight_;
  double bias_;
  double prevBias_;
  int inputsNum_;

  NN7Randomizer* randomGenerator_;
  NN7DataVector* lastInputDataPtr_;

};

#endif
