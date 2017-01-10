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

#include "nn7sigmoidneuron.h"

NN7SigmoidNeuron::NN7SigmoidNeuron(int inputsNum,
  NN7Randomizer* randomGenerator = NULL, double alpha)
  : NN7LinearNeuron(inputsNum, randomGenerator)
{
  alpha_ = alpha;
}

NN7SigmoidNeuron::~NN7SigmoidNeuron()
{
}

NN7UnipolarSigmoidNeuron::NN7UnipolarSigmoidNeuron(int inputsNum,
  NN7Randomizer* randomGenerator, double alpha)
  : NN7SigmoidNeuron(inputsNum, randomGenerator, alpha)
{
}

double NN7UnipolarSigmoidNeuron::getDerivative()
{
  return alpha_ *getLastResponse()*(1 - getLastResponse());
}

double NN7UnipolarSigmoidNeuron::getDerivative(double x)
{
  return alpha_ *sigmoid_function_unipolar(x)*(1 - sigmoid_function_unipolar(x));
}

double NN7UnipolarSigmoidNeuron::response(NN7DataVector* x, double bias)
{
  try
  {
    lastResponse_ = sigmoid_function_unipolar(getV(x) + bias);
  }
  catch (NN7Exception& e)
  {
    RELAY_EXCEPTION(e, "NN7UnipolarSigmoidNeuron", "response",
      "unccorrect input vector");
  }

  return lastResponse_;
}

double NN7UnipolarSigmoidNeuron::sigmoid_function_unipolar(double v)
{
  return 1.0f / (1.0f + std::exp(-alpha_*v));
}

NN7BipolarSigmoidNeuron::NN7BipolarSigmoidNeuron(
  int inputsNum, NN7Randomizer* randomGenerator,
  double alpha) : NN7SigmoidNeuron(inputsNum, randomGenerator, alpha)
{
}

double NN7BipolarSigmoidNeuron::getDerivative()
{
  return alpha_ *getLastResponse()*(1 - getLastResponse());
}

double NN7BipolarSigmoidNeuron::getDerivative(double x)
{
  return alpha_ *sigmoid_function_bipolar(x)*(1 - sigmoid_function_bipolar(x));
}

double NN7BipolarSigmoidNeuron::response(NN7DataVector* x, double bias)
{
  try
  {
    lastResponse_ = sigmoid_function_bipolar(getV(x) + bias);
  }
  catch (NN7Exception& e)
  {
    RELAY_EXCEPTION(e, "NN7BipolarSigmoidNeuron", "response",
      "unccorrect input vector");
  }

  return lastResponse_;
}

double NN7BipolarSigmoidNeuron::sigmoid_function_bipolar(double v)
{
  lastResponse_ = (2.0f / (1.0f + std::exp(-alpha_*v))) - 1;
  return lastResponse_;
}
