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

#ifndef NN7_NN7EXCEPTION_H_
#define NN7_NN7EXCEPTION_H_

#include <iostream>
#include <vector>
#include <string>

#define THROW_EXCEPTION(CLASS_NAME, FUNCTION_NAME, MSG) {\
  NN7Exception::EXCEPTION_REPORT report; \
  report.className = CLASS_NAME; \
  report.functionName = FUNCTION_NAME; \
  report.message = MSG; \
  throw NN7Exception(report); }\

#define RELAY_EXCEPTION(EXCEPTION, CLASS_NAME, FUNCTION_NAME, MSG) {\
  NN7Exception::EXCEPTION_REPORT report; \
  report.className = CLASS_NAME; \
  report.functionName = FUNCTION_NAME; \
  report.message = MSG; \
  EXCEPTION.addExceptionReport(report); \
  throw EXCEPTION; }\


class NN7Exception
{

public:
  struct EXCEPTION_REPORT
  {
    EXCEPTION_REPORT() : exceptionNum(0), errorCode(0) {}
    int exceptionNum;
    std::string className;
    std::string functionName;
    int errorCode;
    std::string message;
  };

  NN7Exception(EXCEPTION_REPORT& report);
  ~NN7Exception();

  void addExceptionReport(EXCEPTION_REPORT& report);
  EXCEPTION_REPORT& lastError();

  static void printLastException(NN7Exception& e);
  static void printAllException(NN7Exception& e);
  static void printReport(EXCEPTION_REPORT& report);

private:
  std::vector<EXCEPTION_REPORT> exceptionStack_;
};

#endif
