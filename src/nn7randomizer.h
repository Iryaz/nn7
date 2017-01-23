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

#ifndef NN7_NN7RANDOMIZER_H_
#define NN7_NN7RANDOMIZER_H_

#include <random>
#include <cmath>
//#include <sys/time.h>

#define RANDOM_DEV_FILE "/dev/urandom"

class NN7Randomizer
{
public:

  NN7Randomizer(double minRange = -1, double maxRange = 1);
  ~NN7Randomizer();

  void setMaxRange(double max) { maxRange_ = max; }
  void setMinRange(double min) { minRange_ = min; }
  void setRange(double min, double max);

  double getMinRange() const { return minRange_; }
  double getMaxRange() const { return maxRange_; }
  double getRandom();

private:
  double minRange_;
  double maxRange_;
};

#endif
