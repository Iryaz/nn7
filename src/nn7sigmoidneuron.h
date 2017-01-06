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

#ifndef NN7_NN7SIGMOIDNEURON_H_
#define NN7_NN7SIGMOIDNEURON_H_

#include <cmath>

#include "nn7linearneuron.h"

class NN7SigmoidNeuron : public NN7LinearNeuron
{
public:

  NN7SigmoidNeuron(int inputsNum, NN7Randomizer* randomGenerator, double alpha = 1.0f);
  ~NN7SigmoidNeuron();

  virtual double response(NN7DataVector* x, double bias = 0) = 0;
  double getAlpha() const { return alpha_; }

  void setAlpha(double alpha) { alpha_ = alpha; }

protected:
  double alpha_;
};

class NN7UnipolarSigmoidNeuron : public NN7SigmoidNeuron
{
public:
  NN7UnipolarSigmoidNeuron(int _inputsNum,
    NN7Randomizer* _randomGenerator = NULL, double _alpha = 1.0f);

  double response(NN7DataVector* x, double bias = 0);
  double getDerivative();

private:
  double sigmoid_function_unipolar(double v);
};

class NN7BipolarSigmoidNeuron : public NN7SigmoidNeuron
{
public:
  NN7BipolarSigmoidNeuron(int _inputsNum,
    NN7Randomizer* _randomGenerator = NULL, double _alpha = 1.0f);

  double response(NN7DataVector* x, double bias = 0);
  double getDerivative();

private:
  double sigmoid_function_bipolar(double v);
};

#endif
