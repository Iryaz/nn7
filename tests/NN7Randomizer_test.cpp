
#include "nn7randomizer.h"
#include <iostream>

#define CATCH_CONFIG_MAIN
#include <catch.hpp>

using std::cout;

#define DIGIT_NUM 100

TEST_CASE("Randomizer test") {
  NN7Randomizer randomizer;

  randomizer.setRange(0, 1);
  cout << "Random number - " << randomizer.getRandom() << " " << "\n";
  REQUIRE(randomizer.getRandom() >= 0);

  randomizer.setRange(-1, 0);
  cout << "Random number - " << randomizer.getRandom() << " " << "\n";
  REQUIRE(randomizer.getRandom() <= 0);

  randomizer.setRange(-1, 1);
  REQUIRE(randomizer.getRandom() >= -1);
  REQUIRE(randomizer.getRandom() <= 1);
}
