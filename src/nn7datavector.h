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

#ifndef NN7_NN7DATAVECTOR_H_
#define NN7_NN7DATAVECTOR_H_

#include <cstddef>
#include <cmath>
#include <initializer_list>

#include "nn7exception.h"

class NN7DataVector
{
public:

  enum NORMALIZE_METHOD {
      EUCLIDEAN,
      MANHATTAN
  };

  NN7DataVector();
  NN7DataVector(uint32_t size, double initVectorValue = 0);
  NN7DataVector(double* valueArray, uint32_t len);
  NN7DataVector(std::initializer_list<double> data);
  NN7DataVector(const NN7DataVector& v);
  ~NN7DataVector();

  void append(double _val);
  void replaceValue(uint32_t index, double newValue);
  void normalize(NORMALIZE_METHOD method = EUCLIDEAN);
  void clear();

  double getValue(uint32_t index) const;
  double operator[](uint32_t index) const;
  NN7DataVector operator-(NN7DataVector& v);
  NN7DataVector operator+(NN7DataVector& v);
  NN7DataVector& operator=(const NN7DataVector& v);

  uint32_t size() const { return elementCount_; }

  static uint32_t vector2Array(NN7DataVector& v, double* arrayPtr, uint32_t arrayLen);

private:

  struct DataVectorElement
  {
    DataVectorElement(double _val = 0.0f) : val(_val), next(NULL), prev(NULL) {}
    double val;
    DataVectorElement *next;
    DataVectorElement *prev;
  };

  DataVectorElement *firstElement_;
  uint32_t elementCount_;

};

#endif
