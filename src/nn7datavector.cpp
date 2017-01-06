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

#include "nn7datavector.h"
#include <iostream>

NN7DataVector::NN7DataVector() : elementCount_(0),
  firstElement_(NULL)
{
}

NN7DataVector::NN7DataVector(int _size, double initVectorValue) :
  elementCount_(0), firstElement_(NULL)
{
  for (int i = 0; i < _size; i++)
    append(initVectorValue);
}

NN7DataVector::NN7DataVector(double* valueArray, int len) :
  elementCount_(0), firstElement_(NULL)
{
  for (int i = 0; i < len; i++)
    append(valueArray[i]);
}

NN7DataVector::NN7DataVector(std::initializer_list<double> data) : elementCount_(0),
  firstElement_(NULL)
{
  std::initializer_list<double>::iterator it;
  for (it = data.begin(); it != data.end(); ++it) {
    append(*it);
  }
}

NN7DataVector::NN7DataVector(const NN7DataVector& v) : elementCount_(0),
  firstElement_(NULL)
{
  DataVectorElement* e = v.firstElement_;

  while (e != NULL) {
    append(e->val);
    e = e->next;
  }
}

NN7DataVector::~NN7DataVector()
{
  clear();
}

void NN7DataVector::clear()
{
  if (firstElement_ != NULL) {
    DataVectorElement* e = firstElement_;
    DataVectorElement* nextElement = NULL;

    for (int i = 0; i < size(); i++) {
      nextElement = e->next;
      delete e;
      e = nextElement;
    }
  }

  elementCount_ = 0;
  firstElement_ = NULL;
}

void NN7DataVector::append(double value)
{

  if (elementCount_ == 0)
    firstElement_ = new DataVectorElement(value);
  else {
    DataVectorElement* e = firstElement_;

    while (e->next != NULL) {
      e = e->next;
    }

    e->next = new DataVectorElement(value);
    e->next->prev = e;
  }

  elementCount_++;
}

void NN7DataVector::replaceValue(int index, double newValue)
{
  if (size() == 0) {
    THROW_EXCEPTION("NN7DataVector", "replaceValue",
      "NN7DataVector::size() == 0");
  }

  if (index > size() - 1) {
    THROW_EXCEPTION("NN7DataVector", "replaceValue",
      "index > NN7DataVector::size()");
  }

  DataVectorElement* e = firstElement_;

  if (index  == 0) {
    e->val = newValue;
  } else {
    int count = 0;

    while (e->next != NULL) {
      count++;
      e = e->next;
      if (index == count)
        break;
    }

    e->val = newValue;
  }
}

double NN7DataVector::getValue(int index) const
{
  if (size() == 0) {
    THROW_EXCEPTION("NN7DataVector", "getValue",
      "NN7DataVector::size() == 0");
  }

  if (index > size() - 1) {
    THROW_EXCEPTION("NN7DataVector", "getValue",
      "index > NN7DataVector::size()");
  }

  if (index == 0)
    return firstElement_->val;

  DataVectorElement* e = firstElement_;

  int count = 0;

  while (e->next != NULL) {
    count++;
    e = e->next;
    if (index == count)
      break;
  }

  return e->val;
}

double NN7DataVector::operator[](int index) const
{
  try
  {
    return getValue(index);
  }
  catch (NN7Exception& e)
  {
    RELAY_EXCEPTION(e, "NN7DataVector", "operator[]",
      "index > NN7DataVector::size()");
  }
}

NN7DataVector NN7DataVector::operator-(NN7DataVector& v)
{
  if (v.size() != size()) {
    THROW_EXCEPTION("NN7DataVector", "operator-",
      "v1::size() != v2::size()");
  }

  NN7DataVector ret;

  for (int i = 0; i < v.size(); i++)
    ret.append(getValue(i) - v.getValue(i));

  return ret;
}

NN7DataVector NN7DataVector::operator+(NN7DataVector& v)
{
  if (v.size() != size()) {
    THROW_EXCEPTION("NN7DataVector", "operator+",
      "v1::size() != v2::size()");
  }

  NN7DataVector ret;

  for (int i = 0; i < v.size(); i++)
    ret.append(getValue(i) + v.getValue(i));

  return ret;
}

NN7DataVector& NN7DataVector::operator=(const NN7DataVector& v)
{
  clear();

  for (int i = 0; i < v.size(); i++)
    append(v[i]);

  return *this;
}

void NN7DataVector::normalize(NORMALIZE_METHOD method)
{
  if (size() == 0) return;

  double strength = 0;

  switch (method) {
    case EUCLIDEAN: {
      for (int i = 0; i < size(); i++)
        strength += pow(getValue(i), 2);
      strength = sqrt(strength);
    } break;

    case MANHATTAN: {
      for (int i = 0; i < size(); i++)
        strength += abs(getValue(i));
    } break;
  }

  if(strength == 0) strength = 1;

  for (int i = 0; i < size(); i++) {
    double currentValue = getValue(i);
    replaceValue(i, currentValue / strength);
  }
}

int NN7DataVector::vector2Array(NN7DataVector& v, double* arrayPtr, int arrayLen)
{
  int copyElementNum = 0;

  if (v.size() > arrayLen)
    return copyElementNum;

  for (int i = 0; i < v.size(); i++) {
    arrayPtr[i] = v[i];
    copyElementNum++;
  }

  return copyElementNum;
}
