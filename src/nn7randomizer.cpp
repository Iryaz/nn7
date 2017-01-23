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
#include <windows.h>

using std::random_device;
using std::uniform_real_distribution;
using std::srand;
using std::rand;

NN7Randomizer::NN7Randomizer(double minRange, double maxRange) :
  minRange_(minRange), maxRange_(maxRange)
{
/*#ifndef _WIN32
	FILE *fp = fopen("/dev/urandom", "r");
	unsigned int foo;
	struct timeval t;

	if(!fp)
	{
	   gettimeofday(&t, NULL);
     foo = t.tv_usec;
	}
	else
	{
    if(fread(&foo, sizeof(foo), 1, fp) != 1)
    {
      gettimeofday(&t, NULL);
      foo = t.tv_usec;
	  }
		fclose(fp);
	}
  srand(foo);
#else
  srand(GetTickCount());
#endif*/
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
  //std::normal_distribution<double> distribution(minRange_, maxRange_);
  return distribution(generator);

  //return ((double)(minRange_))+(((double)(maxRange_)-((double)(minRange_)))*rand()/(1000000000000000.0f+1.0f));
}
