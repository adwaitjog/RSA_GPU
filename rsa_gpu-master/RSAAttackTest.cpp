/* Record the timing of CPU RSA decryption
 * 3/10/2017
 */
#include <boost/program_options.hpp>
namespace po = boost::program_options;
#include <iostream>
#include <iterator>
#include <string>
using namespace std;
#include "RSAAttack.h"

int main(int ac, char* av[]) {
	// Argument parse
	po::options_description desc("Allowed options");
	desc.add_options()
		("help", "produce help message")
		("traceNum", po::value<int>(), "set traceNum")
		("timeFile", po::value<string>(), "file name for time data")
		("swType", po::value<int>(), "0: none; 1: clnw; 3 vlnw")
		("windowSize", po::value<int>(), "CRT attack window size")
		("time", po::value<char>(), "Y: record time,  N: no timing")
		("attack", po::value<char>(), "Y: do attack, N: no attack")
		("attackCRT", po::value<char>(), "Y: do CRT attack, N: no CRT attack")
		("timeCRT", po::value<char>(), "Y: yes, N: no")
		("attackCRTRandom", po::value<char>(), "Y: yes, N: no")
		;
	po::variables_map vm;
	po::store(po::parse_command_line(ac, av, desc), vm);
	po::notify(vm);

	int traceNum = 1024;
	string timeFile("data_cpu.bin");
	SW_Type swType = SW_Type::clnw;
	int windowSize = 16;
	bool time = false;
	bool attack = false;
	bool attackCRT = false;
	bool timeCRT = false;
	bool attackCRTRandom = false;

	if (vm.count("help")) { cout << desc << endl; return 0; }
	if (vm.count("traceNum"))   { traceNum   = vm["traceNum"].as<int>(); }
	if (vm.count("timeFile"))   { timeFile   = vm["timeFile"].as<string>(); }
	if (vm.count("windowSize")) { windowSize = vm["windowSize"].as<int>(); }
	if (vm.count("time")) {
		if (vm["time"].as<char>() == 'Y' || vm["time"].as<char>() == 'y')
			time = true;
	}
	if (vm.count("attack")) {
		if (vm["attack"].as<char>() == 'Y' || vm["attack"].as<char>() == 'y')
			attack = true;
	}
	if (vm.count("attackCRT")) {
		if (vm["attackCRT"].as<char>() == 'Y' || vm["attackCRT"].as<char>() == 'y')
			attackCRT = true;
	}
	if (vm.count("timeCRT")) {
		if (vm["timeCRT"].as<char>() == 'Y' || vm["timeCRT"].as<char>() == 'y')
			timeCRT = true;
	}
	if (vm.count("attackCRTRandom")) {
		if (vm["attackCRTRandom"].as<char>() == 'Y' || vm["attackCRTRandom"].as<char>() == 'y')
			attackCRTRandom = true;
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

	// Display options
	cout << "traceName: "       << traceNum        << endl;
	cout << "timeFile: "        << timeFile        << endl;
	cout << "swType: "           << swType          << endl;
	cout << "windowSize "       << windowSize      << endl;
	cout << "time: "            << time            << endl;
	cout << "attack: "          << attack          << endl;
	cout << "attackCRT: "       << attackCRT       << endl;
	cout << "timeCRT: "         << timeCRT         << endl;
	cout << "attackCRTRandom: " << attackCRTRandom << endl;

	RSAAttack rsaAttack(traceNum, 0, swType);
	// Record the time of decryption of traceNum msges
	if (time) {
		cout << "Timing RSA ..." << endl;
		rsaAttack.timeDecrypt(timeFile.c_str());
	}
	if (attack) {
		cout << "Attacking RSA ..." << endl;
		rsaAttack.timingAttack(timeFile.c_str());
	}
	if (attackCRT) {
		cout << "Attacking CRT ..." << endl;
		rsaAttack.attackCRT(windowSize);
	}
	if (timeCRT) {
		cout << "Timing CRT ..." << endl;
		rsaAttack.timeCRT(timeFile.c_str());
	}
	if (attackCRTRandom) {
		cout << "Attacking CRT Random..." << endl;
		rsaAttack.attackCRTRandom(timeFile.c_str());
	}
}
