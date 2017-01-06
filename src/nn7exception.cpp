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

#include "nn7exception.h"

using std::cout;

NN7Exception::NN7Exception(EXCEPTION_REPORT& report)
{
  addExceptionReport(report);
}

NN7Exception::~NN7Exception()
{

}

void NN7Exception::addExceptionReport(EXCEPTION_REPORT& report)
{
  report.exceptionNum = exceptionStack_.size();
  exceptionStack_.push_back(report);
}

NN7Exception::EXCEPTION_REPORT& NN7Exception::lastError()
{
  return exceptionStack_.back();
}

void NN7Exception::printReport(EXCEPTION_REPORT& report)
{
  cout << "\n" << "Exception " << report.exceptionNum << ": "
    << report.className
    << "::" << report.functionName << " ("
    << report.message << ")" << "\n";
}

void NN7Exception::printLastException(NN7Exception& e)
{
  NN7Exception::printReport(e.lastError());
}

void NN7Exception::printAllException(NN7Exception& e)
{
  cout << "\n====== Begin Exceptions stack -> \n";
  for (int i = 0; i < e.exceptionStack_.size(); i++) {
    NN7Exception::printReport(e.exceptionStack_[i]);
  }

  cout << "\n-> End stack\n";
}
