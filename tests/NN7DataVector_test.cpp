
#include <iostream>

#include "nn7datavector.h"

#define CATCH_CONFIG_MAIN
#include <catch.hpp>

using std::cout;

void printVector(NN7DataVector &v)
{
  for(int i = 0; i < v.size(); i++)
    cout << v[i] << " ";
  cout << "\n";
}

void testCopyObj(NN7DataVector v)
{
  for(int i = 0; i < v.size(); i++)
    cout << v[i] << " ";
  cout << "\n";
}

TEST_CASE("Data vector test") {
  NN7DataVector vector {1, 3, 66};
  cout << "Vector element: ";
  printVector(vector);

  vector.append(9.96);
  cout << "Vector new elements ";
  printVector(vector);

  SECTION("Copy constructor test") {
    testCopyObj(vector);
  }

  SECTION("std array constructor test") {
    double testArray[6] = {45.2, 1, 66, 66.66, 66.4, 77.8};
    NN7DataVector arrayVector(&testArray[0], 6);
    cout << "NN7DataVector arg = double array\n";
    printVector(arrayVector);
  }

  SECTION("Test Init NN7DataVector") {
    NN7DataVector v(97, 4.5);
    cout << "NN7DataVector arg = 4.5 size = 67\n";
    printVector(v);
  }

  SECTION("vector2Array function test") {
    double *outArray = new double[4];
    NN7DataVector::vector2Array(vector, &outArray[0], 4);
    cout << "Double array: ";
    for (int i = 0; i < 4; i++)
      cout << outArray[i] << " ";
    cout << "\n";
  }

  SECTION("Check exception") {
    NN7DataVector data { 1, 34, 342, 123 };
    try {
      data.getValue(data.size() + 1);
    }
    catch (NN7Exception &e)
    {
        NN7Exception::printLastException(e);
        cout << "\nUncorrect index element - test success!\n";
    }

    cout << "Clear vector!\n";
    data.clear();

    try {
      data.getValue(0);
    }
    catch (NN7Exception &e)
    {
      //if (e.getErrorCode() == (int)DataVectorException::EMPTY_DATA_VECTOR)
      NN7Exception::printLastException(e);
      cout << "\nEmpty data vector - test success\n";
    }
  }

  SECTION("Test replace value") {
    try {
      NN7DataVector data { 1, 34, 342, 123 };
      cout << "Data vector: ";
      printVector(data);

      cout << "Replace value - index 0\n";
      data.replaceValue(0, 144);
      cout << "New vector: ";
      printVector(data);
      cout << "\n";

      cout << "Replace value - index 3\n";
      data.replaceValue(data.size()-1, 2556.67);
      cout << "New vector: ";
      printVector(data);
      cout << "\n";

      cout << "Replace value - index 5\n";
      data.replaceValue(5, 2);
      cout << "New vector: ";
      printVector(data);
      cout << "\n";
    }
    catch(NN7Exception &e)
    {
      //if (e.getErrorCode() == DataVectorException::UNCORRECT_INDEX_ELEMENT) {
      NN7Exception::printLastException(e);
      cout << "\nError replace element (uncorrect index) - test success!!!\n";

    }
  }

  SECTION("Test normalize vector") {
    cout << "Random vector: ";
    NN7DataVector sourceVector { 234, 23423, 23111, 111, 7 };
    printVector(sourceVector);
    cout << "Vector normalized (EUCLIDEAN method): ";
    NN7DataVector euclideanNormalizedVector = sourceVector;
    euclideanNormalizedVector.normalize();
    printVector(euclideanNormalizedVector);

    cout << "Vector normalized (MANHATTAN method): ";
    NN7DataVector manhattanNormalizedVector = sourceVector;
    manhattanNormalizedVector.normalize(NN7DataVector::MANHATTAN);
    printVector(manhattanNormalizedVector);
    cout << "\n";
  }

  SECTION("Test operator +/-") {
    NN7DataVector vector1 { 1, 2, 3 };
    NN7DataVector vector2 { 45, 1, 0 };
    NN7DataVector vector3 = vector1 - vector2;
    cout << "{ 1, 2, 3 } - { 45, 1, 0 } = ";
    printVector(vector3);
    cout << "\n";

    vector3 = vector1 + vector2;
    cout << "{ 1, 2, 3 } + { 45, 1, 0 } = ";
    printVector(vector3);
    cout << "\n";
  }
}
