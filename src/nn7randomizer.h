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

class NN7Randomizer
{
public:

  NN7Randomizer(double _minRange = -1, double _maxRange = 1);
  ~NN7Randomizer();

  void setMaxRange(double _max) { maxRange = _max; }
  void setMinRange(double _min) { minRange = _min; }
  void setRange(double _min, double _max);

  double getMinRange() const { return minRange; }
  double getMaxRange() const { return maxRange; }
  double getRandom();

private:
  double minRange;
  double maxRange;
};

#endif
