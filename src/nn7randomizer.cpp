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

#include "nn7randomizer.h"

using std::random_device;
using std::uniform_real_distribution;

NN7Randomizer::NN7Randomizer(double minRange, double maxRange) :
  minRange_(minRange), maxRange_(maxRange)
{
}

NN7Randomizer::~NN7Randomizer()
{

}

void NN7Randomizer::setRange(double min, double max) {
  minRange_ = min;
  maxRange_ = max;
}

double NN7Randomizer::getRandom()
{
  random_device generator(RANDOM_DEV_FILE);
  uniform_real_distribution<double> distribution(minRange_, maxRange_);
  return distribution(generator);
}
