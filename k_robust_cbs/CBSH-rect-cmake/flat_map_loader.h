// Load's a 2D map.
#pragma once

#include <string>
#include <vector>
#include "map_loader.h"
#ifdef NO_PYTHON
// Stub definitions for CLI build (Python not needed)
#ifndef BOOST_PYTHON_STUB_DEFINED
#define BOOST_PYTHON_STUB_DEFINED
namespace boost {
namespace python {
class object {};
}
}
namespace p = boost::python;
#endif
#else
#include <boost/python.hpp>
namespace p = boost::python;
#endif
using namespace std;

struct railCell {
	int transitions;
	bool isTurn;
	bool isDeadEnd;
};

class FlatlandLoader:public MapLoader {
public:
#ifdef NO_PYTHON
	FlatlandLoader() {}  // Default constructor for CLI
	FlatlandLoader(void* rail1, int rows, int cols) {}  // Stub for CLI
#else
	FlatlandLoader(boost::python::object rail1, int rows, int cols);
#endif
	railCell get_full_cell(int location);
	int getDegree(int loc);
	int getDegree(int loc,int heading) const;
    bool notCorner(int loc){
        return true;
    };

#ifdef NO_PYTHON
    void* rail;  // Stub for CLI
#else
    boost::python::object rail;
#endif
	railCell* railMap;
	vector<pair<int, int>> get_transitions(int location, int heading = -1, bool noWait=false) const;
	~FlatlandLoader();
protected:
	float blockRate;
};

