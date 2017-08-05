/* RSAGPUAttack test
 * Time the decryption and record the reduction
 * 3/10/2017
 */
#include <iostream>
#include <boost/program_options.hpp>
namespace po = boost::program_options;
#include <iterator>
#include <string>

using namespace std;

#include "RSAGPUAttack.h"

int main (int argc, char *argv[]) {
	// Argument parse
	po::options_description desc("Time GPU RSA and record reduction");
	desc.add_options()
		("help", "produce help message")
		("traceNum", po::value<int>(), "set traceNum, default 1024")
		("traceSize", po::value<int>(), "set traceSize, default 1")
		("timeFile", po::value<string>(), "file name for timing data, default data2.bin")
		("reductionFile", po::value<string>(), "file name for reduction data, default reduction.bin")
		("seed", po::value<int>(), "set the seed of random number generator")
		("swType", po::value<int>(), "0: none; 1: clnw; 2 vlnw")
		("reduction", po::value<char>(), "Y record reduction, N skip it")
		("time", po::value<char>(), "Y record timing, N skip timing")
		("attack", po::value<char>(), "Y attack, N skip attack")
		;
	po::variables_map vm;
	po::store(po::parse_command_line(argc, argv, desc), vm);
	po::notify(vm);

	// Set arguments: traceNum, timeFileName, reductionFileName;
	int traceNum = 1024;
	int traceSize = 1;
	string timeFile("data_gpu.bin");
	string reductionFile("reduction.bin");
	int seed = 0;
	SW_Type swType = SW_Type::none;
	bool reduction = false;
	bool time = false;
	bool attack = false;

	if (vm.count("help")) {
		cout << desc << endl;
		return 0;
	}
	if (vm.count("traceNum"))      traceNum      = vm["traceNum"].as<int>();
	if (vm.count("traceSize"))     traceSize     = vm["traceSize"].as<int>();
	if (vm.count("timeFile"))      timeFile      = vm["timeFile"].as<string>();
	if (vm.count("reductionFile")) reductionFile = vm["reductionFile"].as<string>();
	if (vm.count("seed"))          seed          = vm["seed"].as<int>();
	if (vm.count("reduction")) {
		if (vm["reduction"].as<char>() == 'Y' || vm["t"].as<char>() == 'y')
			reduction = true;
	}
	if (vm.count("time")) {
		if (vm["time"].as<char>() == 'Y' || vm["time"].as<char>() == 'y')
			time = true;
	}
	if (vm.count("attack")) {
		if (vm["attack"].as<char>() == 'Y' || vm["t"].as<char>() == 'y')
			attack = true;
	}
	if (vm.count("swType")) {
		switch (vm["swType"].as<int>()) {
		case 0:
			swType = SW_Type::none;
			break;
		case 1:
			swType = SW_Type::clnw;
			break;
		case 2:
			swType = SW_Type::vlnw;
			break;
		default:
			throw runtime_error("Unsupported SW_Type");
		}
	}

	// Print argument
	cout << "traceNum = "      << traceNum      << endl
	     << "traceSize = "     << traceSize     << endl
	     << "timeFile = "      << timeFile      << endl
		 << "reductionFile = " << reductionFile << endl
		 << "seed = "          << seed          << endl
		 << "swType = "        << swType        << endl
		 << "reduction = "     << reduction     << endl
		 << "time = "          << time          << endl
		 << "attack = "        << attack        << endl
		 << endl;

    RSAGPUAttack rsaGPUAttack(traceNum, traceSize, 0, swType);
	if (reduction) {
		cout << "Recording reduction ..." << endl;
		rsaGPUAttack.recordReduction(reductionFile.c_str());
	}
	if (time) { 
		cout << "Recording time ..." << endl;
		rsaGPUAttack.recordTime(timeFile.c_str());
	}
	if (attack) {
		cout << "Attacking ..." << endl;
		rsaGPUAttack.timingAttack(timeFile.c_str());
	}
}
