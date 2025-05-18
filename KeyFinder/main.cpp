#include <stdio.h>
#include <fstream>
#include <iostream>

#include "KeyFinder.h"
#include "AddressUtil.h"
#include "util.h"
#include "secp256k1.h"
#include "CmdParse.h"
#include "Logger.h"
#include "ConfigFile.h"
#include <random>

#include "DeviceManager.h"

#ifdef BUILD_CUDA
#include "CudaKeySearchDevice.h"
#endif

#ifdef BUILD_OPENCL
#include "CLKeySearchDevice.h"
#endif

typedef struct {
    // startKey is the first key. We store it so that if the --continue
    // option is used, the correct progress is displayed. startKey and
    // nextKey are only equal at the very beginning. nextKey gets saved
    // in the checkpoint file.
    secp256k1::uint256 startKey = 1;
    secp256k1::uint256 nextKey = 1;

    // The last key to be checked
    secp256k1::uint256 endKey = secp256k1::N - 1;

    uint64_t statusInterval = 1800;
    uint64_t checkpointInterval = 60000;

    unsigned int threads = 0;
    unsigned int blocks = 0;
    unsigned int pointsPerThread = 0;
    
    int compression = PointCompressionType::COMPRESSED;
 
    std::vector<std::string> targets;

    std::string targetsFile = "";

    std::string checkpointFile = "";

    int device = 0;

    std::string resultsFile = "";

    uint64_t totalkeys = 0;
    unsigned int elapsed = 0;
    secp256k1::uint256 stride = 1;

    bool follow = false;
}RunConfig;

static RunConfig _config;

std::vector<DeviceManager::DeviceInfo> _devices;

void writeCheckpoint(secp256k1::uint256 nextKey);

static uint64_t _lastUpdate = 0;
static uint64_t _runningTime = 0;
static uint64_t _startTime = 0;
static bool _rangeMode = false;
static size_t _currentRangeIdx = 0;
static size_t _rangesRemaining = 0;
static secp256k1::uint256 _currentRangeEnd = 0;
static size_t _totalRanges = 0;

/**
* Callback to display the private key
*/
void resultCallback(KeySearchResult info)
{
	if(_config.resultsFile.length() != 0) {
		Logger::log(LogLevel::Info, "Found key for address '" + info.address + "'. Written to '" + _config.resultsFile + "'");

		std::string s = info.address + " " + info.privateKey.toString(16) + " " + info.publicKey.toString(info.compressed);
		util::appendToFile(_config.resultsFile, s);

		return;
	}

	std::string logStr = "Address:     " + info.address + "\n";
	logStr += "Private key: " + info.privateKey.toString(16) + "\n";
	logStr += "Compressed:  ";

	if(info.compressed) {
		logStr += "yes\n";
	} else {
		logStr += "no\n";
	}

	logStr += "Public key:  \n";

	if(info.compressed) {
		logStr += info.publicKey.toString(true) + "\n";
	} else {
		logStr += info.publicKey.x.toString(16) + "\n";
		logStr += info.publicKey.y.toString(16) + "\n";
	}

	Logger::log(LogLevel::Info, logStr);
}

/**
Callback to display progress
*/
void statusCallback(KeySearchStatus info)
{
	std::string speedStr;

	if(info.speed < 0.01) {
		speedStr = "< 0.01 MKey/s";
	} else {
		speedStr = util::format("%.2f", info.speed) + " MKey/s";
	}

	std::string totalStr = "(" + util::formatThousands(_config.totalkeys + info.total) + " total)";

	std::string timeStr = "[" + util::formatSeconds((unsigned int)((_config.elapsed + info.totalTime) / 1000)) + "]";

	std::string usedMemStr = util::format((info.deviceMemory - info.freeMemory) /(1024 * 1024));

	std::string totalMemStr = util::format(info.deviceMemory / (1024 * 1024));

    std::string targetStr = util::format(info.targets) + " target" + (info.targets > 1 ? "s" : "");


	// Fit device name in 16 characters, pad with spaces if less
	std::string devName = info.deviceName.substr(0, 16);
	devName += std::string(16 - devName.length(), ' ');

    const char *formatStr = NULL;

    if(_config.follow) {
        formatStr = "%s %s/%sMB | %s %s %s %s";
    } else {
        formatStr = "\r%s %s / %sMB | %s %s %s %s";
    }

    printf(formatStr, devName.c_str(), usedMemStr.c_str(), totalMemStr.c_str(), targetStr.c_str(), speedStr.c_str(), totalStr.c_str(), timeStr.c_str());

    if(_rangeMode) {
        secp256k1::uint256 remaining = _currentRangeEnd - info.nextKey;
        uint64_t rem = remaining.toUint64() + 1;
        double etaSec = 0.0;
        if(info.speed > 0.0) {
            etaSec = (double)rem / (info.speed * 1000000.0);
        }
        std::string etaStr = util::formatSeconds((unsigned int)etaSec);
        std::string rangeInfo = " | range " + util::format((int)(_currentRangeIdx + 1)) + "/" + util::format((int)_totalRanges);
        rangeInfo += " remaining:" + util::format((int)_rangesRemaining) + " eta:" + etaStr;
        printf("%s", rangeInfo.c_str());
    }

    if(_config.follow) {
        printf("\n");
    }

    if(_config.checkpointFile.length() > 0) {
        uint64_t t = util::getSystemTime();
        if(t - _lastUpdate >= _config.checkpointInterval) {
            Logger::log(LogLevel::Info, "Checkpoint");
            writeCheckpoint(info.nextKey);
            _lastUpdate = t;
        }
    }
}

/**
 * Parses the start:end key pair. Possible values are:
 start
 start:end
 start:+offset
 :end
 :+offset
 */
bool parseKeyspace(const std::string &s, secp256k1::uint256 &start, secp256k1::uint256 &end)
{
    size_t pos = s.find(':');

    if(pos == std::string::npos) {
        start = secp256k1::uint256(s);
        end = secp256k1::N - 1;
    } else {
        std::string left = s.substr(0, pos);

        if(left.length() == 0) {
            start = secp256k1::uint256(1);
        } else {
            start = secp256k1::uint256(left);
        }

        std::string right = s.substr(pos + 1);

        if(right[0] == '+') {
            end = start + secp256k1::uint256(right.substr(1));
        } else {
            end = secp256k1::uint256(right);
        }
    }

    return true;
}

void usage()
{
    printf("BitCrack OPTIONS [TARGETS]\n");
    printf("Where TARGETS is one or more addresses\n\n");
	
    printf("--help                  Display this message\n");
    printf("-c, --compressed        Use compressed points\n");
    printf("-u, --uncompressed      Use Uncompressed points\n");
    printf("--compression  MODE     Specify compression where MODE is\n");
    printf("                          COMPRESSED or UNCOMPRESSED or BOTH\n");
    printf("-d, --device ID         Use device ID\n");
    printf("-b, --blocks N          N blocks\n");
    printf("-t, --threads N         N threads per block\n");
    printf("-p, --points N          N points per thread\n");
    printf("-i, --in FILE           Read addresses from FILE, one per line\n");
    printf("-o, --out FILE          Write keys to FILE\n");
    printf("-f, --follow            Follow text output\n");
    printf("--list-devices          List available devices\n");
    printf("--keyspace KEYSPACE     Specify the keyspace:\n");
    printf("                          START:END\n");
    printf("                          START:+COUNT\n");
    printf("                          START\n");
    printf("                          :END\n"); 
    printf("                          :+COUNT\n");
    printf("                        Where START, END, COUNT are in hex format\n");
    printf("--stride N              Increment by N keys at a time\n");
    printf("--share M/N             Divide the keyspace into N equal shares, process the Mth share\n");
    printf("--continue FILE         Save/load progress from FILE\n");
    printf("--create-ranges FILE    Create ranges covering the keyspace\n");
    printf("--process-ranges FILE   Process ranges from FILE\n");
}


/**
 Finds default parameters depending on the device
 */
typedef struct {
	int threads;
	int blocks;
	int pointsPerThread;
}DeviceParameters;

DeviceParameters getDefaultParameters(const DeviceManager::DeviceInfo &device)
{
	DeviceParameters p;
	p.threads = 256;
    p.blocks = 32;
	p.pointsPerThread = 32;

	return p;
}

static KeySearchDevice *getDeviceContext(DeviceManager::DeviceInfo &device, int blocks, int threads, int pointsPerThread)
{
#ifdef BUILD_CUDA
    if(device.type == DeviceManager::DeviceType::CUDA) {
        return new CudaKeySearchDevice((int)device.physicalId, threads, pointsPerThread, blocks);
    }
#endif

#ifdef BUILD_OPENCL
    if(device.type == DeviceManager::DeviceType::OpenCL) {
        return new CLKeySearchDevice(device.physicalId, threads, pointsPerThread, blocks);
    }
#endif

    return NULL;
}

static void printDeviceList(const std::vector<DeviceManager::DeviceInfo> &devices)
{
    for(int i = 0; i < devices.size(); i++) {
        printf("ID:     %d\n", devices[i].id);
        printf("Name:   %s\n", devices[i].name.c_str());
        printf("Memory: %lluMB\n", (unsigned long long)(devices[i].memory / ((uint64_t)1024 * 1024)));
        printf("Compute units: %d\n", devices[i].computeUnits);
        printf("\n");
    }
}

bool readAddressesFromFile(const std::string &fileName, std::vector<std::string> &lines)
{
    if(fileName == "-") {
        return util::readLinesFromStream(std::cin, lines);
    } else {
        return util::readLinesFromStream(fileName, lines);
    }
}

int parseCompressionString(const std::string &s)
{
    std::string comp = util::toLower(s);

    if(comp == "both") {
        return PointCompressionType::BOTH;
    }

    if(comp == "compressed") {
        return PointCompressionType::COMPRESSED;
    }

    if(comp == "uncompressed") {
        return PointCompressionType::UNCOMPRESSED;
    }

    throw std::string("Invalid compression format: '" + s + "'");
}

static std::string getCompressionString(int mode)
{
    switch(mode) {
    case PointCompressionType::BOTH:
        return "both";
    case PointCompressionType::UNCOMPRESSED:
        return "uncompressed";
    case PointCompressionType::COMPRESSED:
        return "compressed";
    }

    throw std::string("Invalid compression setting '" + util::format(mode) + "'");
}

void writeCheckpoint(secp256k1::uint256 nextKey)
{
    std::ofstream tmp(_config.checkpointFile, std::ios::out);

    tmp << "start=" << _config.startKey.toString() << std::endl;
    tmp << "next=" << nextKey.toString() << std::endl;
    tmp << "end=" << _config.endKey.toString() << std::endl;
    tmp << "blocks=" << _config.blocks << std::endl;
    tmp << "threads=" << _config.threads << std::endl;
    tmp << "points=" << _config.pointsPerThread << std::endl;
    tmp << "compression=" << getCompressionString(_config.compression) << std::endl;
    tmp << "device=" << _config.device << std::endl;
    tmp << "elapsed=" << (_config.elapsed + util::getSystemTime() - _startTime) << std::endl;
    tmp << "stride=" << _config.stride.toString();
    tmp.close();
}

void readCheckpointFile()
{
    if(_config.checkpointFile.length() == 0) {
        return;
    }

    ConfigFileReader reader(_config.checkpointFile);

    if(!reader.exists()) {
        return;
    }

    Logger::log(LogLevel::Info, "Loading ' " + _config.checkpointFile + "'");

    std::map<std::string, ConfigFileEntry> entries = reader.read();

    _config.startKey = secp256k1::uint256(entries["start"].value);
    _config.nextKey = secp256k1::uint256(entries["next"].value);
    _config.endKey = secp256k1::uint256(entries["end"].value);

    if(_config.threads == 0 && entries.find("threads") != entries.end()) {
        _config.threads = util::parseUInt32(entries["threads"].value);
    }
    if(_config.blocks == 0 && entries.find("blocks") != entries.end()) {
        _config.blocks = util::parseUInt32(entries["blocks"].value);
    }
    if(_config.pointsPerThread == 0 && entries.find("points") != entries.end()) {
        _config.pointsPerThread = util::parseUInt32(entries["points"].value);
    }
    if(entries.find("compression") != entries.end()) {
        _config.compression = parseCompressionString(entries["compression"].value);
    }
    if(entries.find("elapsed") != entries.end()) {
        _config.elapsed = util::parseUInt32(entries["elapsed"].value);
    }
    if(entries.find("stride") != entries.end()) {
        _config.stride = util::parseUInt64(entries["stride"].value);
    }

    _config.totalkeys = (_config.nextKey - _config.startKey).toUint64();
}

int run()
{
    if(_config.device < 0 || _config.device >= _devices.size()) {
        Logger::log(LogLevel::Error, "device " + util::format(_config.device) + " does not exist");
        return 1;
    }

    Logger::log(LogLevel::Info, "Compression: " + getCompressionString(_config.compression));
    Logger::log(LogLevel::Info, "Starting at: " + _config.nextKey.toString());
    Logger::log(LogLevel::Info, "Ending at:   " + _config.endKey.toString());
    Logger::log(LogLevel::Info, "Counting by: " + _config.stride.toString());

    try {

        _lastUpdate = util::getSystemTime();
        _startTime = util::getSystemTime();

        // Use default parameters if they have not been set
        DeviceParameters params = getDefaultParameters(_devices[_config.device]);

        if(_config.blocks == 0) {
            _config.blocks = params.blocks;
        }

        if(_config.threads == 0) {
            _config.threads = params.threads;
        }

        if(_config.pointsPerThread == 0) {
            _config.pointsPerThread = params.pointsPerThread;
        }

        // Get device context
        KeySearchDevice *d = getDeviceContext(_devices[_config.device], _config.blocks, _config.threads, _config.pointsPerThread);

        KeyFinder f(_config.nextKey, _config.endKey, _config.compression, d, _config.stride);

        f.setResultCallback(resultCallback);
        f.setStatusInterval(_config.statusInterval);
        f.setStatusCallback(statusCallback);

        f.init();

        if(!_config.targetsFile.empty()) {
            f.setTargets(_config.targetsFile);
        } else {
            f.setTargets(_config.targets);
        }

        f.run();

        delete d;
    } catch(KeySearchException ex) {
        Logger::log(LogLevel::Info, "Error: " + ex.msg);
        return 1;
    }

    return 0;
}

/**
 * Parses a string in the form of x/y
 */
bool parseShare(const std::string &s, uint32_t &idx, uint32_t &total)
{
    size_t pos = s.find('/');
    if(pos == std::string::npos) {
        return false;
    }

    try {
        idx = util::parseUInt32(s.substr(0, pos));
    } catch(...) {
        return false;
    }

    try {
        total = util::parseUInt32(s.substr(pos + 1));
    } catch(...) {
        return false;
    }

    if(idx == 0 || total == 0) {
        return false;
    }

    if(idx > total) {
        return false;
    }

    return true;
}

struct RangeEntry {
    secp256k1::uint256 start;
    secp256k1::uint256 end;
    bool done = false;
};

static bool loadRanges(const std::string &file, std::vector<RangeEntry> &ranges)
{
    ranges.clear();
    std::vector<std::string> lines;
    if(!util::readLinesFromStream(file, lines)) {
        return false;
    }

    for(size_t i = 0; i < lines.size(); i++) {
        std::string line = util::trim(lines[i]);
        if(line.length() == 0) continue;
        size_t p1 = line.find(':');
        if(p1 == std::string::npos) continue;
        size_t p2 = line.find(':', p1 + 1);
        std::string s1 = line.substr(0, p1);
        std::string s2 = (p2 == std::string::npos) ? line.substr(p1 + 1) : line.substr(p1 + 1, p2 - p1 - 1);
        std::string s3 = (p2 == std::string::npos) ? "0" : line.substr(p2 + 1);
        RangeEntry r;
        r.start = secp256k1::uint256(s1);
        r.end = secp256k1::uint256(s2);
        r.done = (s3 == "1");
        ranges.push_back(r);
    }

    return true;
}

static bool saveRanges(const std::string &file, const std::vector<RangeEntry> &ranges)
{
    std::ofstream out(file.c_str(), std::ios::out);
    if(!out.is_open()) {
        return false;
    }

    for(size_t i = 0; i < ranges.size(); i++) {
        out << ranges[i].start.toString() << ":" << ranges[i].end.toString() << ":" << (ranges[i].done ? "1" : "0") << std::endl;
    }

    out.close();
    return true;
}

static void createRangesFile(const std::string &file)
{
    std::ofstream out(file.c_str(), std::ios::out);
    if(!out.is_open()) {
        Logger::log(LogLevel::Error, "Unable to write '" + file + "'");
        return;
    }

    secp256k1::uint256 rangeSize(static_cast<uint64_t>(300ULL * 1000000ULL * 60ULL * 60ULL));
    secp256k1::uint256 cur = _config.startKey;
    size_t count = 0;

    while(cur.cmp(_config.endKey) <= 0) {
        secp256k1::uint256 end = cur + rangeSize - 1;
        if(end.cmp(_config.endKey) > 0) {
            end = _config.endKey;
        }

        out << cur.toString() << ":" << end.toString() << ":0" << std::endl;
        count++;

        if(end.cmp(_config.endKey) == 0) {
            break;
        }

        cur = end + 1;
    }

    out.close();
    Logger::log(LogLevel::Info, util::format((int)count) + " ranges written to '" + file + "'");
}

static int processRanges(const std::string &file)
{
    std::vector<RangeEntry> ranges;
    if(!loadRanges(file, ranges)) {
        Logger::log(LogLevel::Error, "Unable to read '" + file + "'");
        return 1;
    }

    std::random_device rd;
    std::mt19937_64 gen(rd());

    while(true) {
        std::vector<size_t> remaining;
        for(size_t i = 0; i < ranges.size(); i++) {
            if(!ranges[i].done) remaining.push_back(i);
        }

        if(remaining.size() == 0) {
            Logger::log(LogLevel::Info, "All ranges processed");
            break;
        }

        std::uniform_int_distribution<size_t> dist(0, remaining.size() - 1);
        size_t idx = remaining[dist(gen)];
        RangeEntry &r = ranges[idx];

        _rangeMode = true;
        _currentRangeIdx = idx;
        _rangesRemaining = remaining.size() - 1;
        _totalRanges = ranges.size();
        _currentRangeEnd = r.end;

        _config.startKey = r.start;
        _config.nextKey = r.start;
        _config.endKey = r.end;
        _config.totalkeys = 0;
        _config.elapsed = 0;

        int rc = run();
        if(rc != 0) {
            return rc;
        }

        r.done = true;
        saveRanges(file, ranges);
        _rangeMode = false;
    }

    return 0;
}

int main(int argc, char **argv)
{
	bool optCompressed = false;
	bool optUncompressed = false;
    bool listDevices = false;
    bool optShares = false;
    bool optThreads = false;
    bool optBlocks = false;
    bool optPoints = false;
    bool optCreateRanges = false;
    bool optProcessRanges = false;
    std::string rangesCreateFile = "";
    std::string rangesProcessFile = "";

    uint32_t shareIdx = 0;
    uint32_t numShares = 0;

    // Catch --help first
    for(int i = 1; i < argc; i++) {
        if(std::string(argv[i]) == "--help") {
            usage();
            return 0;
        }
    }

    // Check for supported devices
    try {
        _devices = DeviceManager::getDevices();

        if(_devices.size() == 0) {
            Logger::log(LogLevel::Error, "No devices available");
            return 1;
        }
    } catch(DeviceManager::DeviceManagerException ex) {
        Logger::log(LogLevel::Error, "Error detecting devices: " + ex.msg);
        return 1;
    }

    // Check for arguments
	if(argc == 1) {
		usage();
		return 0;
	}


	CmdParse parser;
	parser.add("-d", "--device", true);
	parser.add("-t", "--threads", true);
	parser.add("-b", "--blocks", true);
	parser.add("-p", "--points", true);
	parser.add("-d", "--device", true);
	parser.add("-c", "--compressed", false);
	parser.add("-u", "--uncompressed", false);
    parser.add("", "--compression", true);
	parser.add("-i", "--in", true);
	parser.add("-o", "--out", true);
    parser.add("-f", "--follow", false);
    parser.add("", "--list-devices", false);
    parser.add("", "--keyspace", true);
    parser.add("", "--continue", true);
    parser.add("", "--share", true);
    parser.add("", "--stride", true);
    parser.add("", "--create-ranges", true);
    parser.add("", "--process-ranges", true);

    try {
        parser.parse(argc, argv);
    } catch(std::string err) {
        Logger::log(LogLevel::Error, "Error: " + err);
        return 1;
    }

    std::vector<OptArg> args = parser.getArgs();

	for(unsigned int i = 0; i < args.size(); i++) {
		OptArg optArg = args[i];
		std::string opt = args[i].option;

		try {
			if(optArg.equals("-t", "--threads")) {
				_config.threads = util::parseUInt32(optArg.arg);
                optThreads = true;
            } else if(optArg.equals("-b", "--blocks")) {
                _config.blocks = util::parseUInt32(optArg.arg);
                optBlocks = true;
			} else if(optArg.equals("-p", "--points")) {
				_config.pointsPerThread = util::parseUInt32(optArg.arg);
                optPoints = true;
			} else if(optArg.equals("-d", "--device")) {
				_config.device = util::parseUInt32(optArg.arg);
			} else if(optArg.equals("-c", "--compressed")) {
				optCompressed = true;
            } else if(optArg.equals("-u", "--uncompressed")) {
                optUncompressed = true;
            } else if(optArg.equals("", "--compression")) {
                _config.compression = parseCompressionString(optArg.arg);
			} else if(optArg.equals("-i", "--in")) {
				_config.targetsFile = optArg.arg;
			} else if(optArg.equals("-o", "--out")) {
				_config.resultsFile = optArg.arg;
            } else if(optArg.equals("", "--list-devices")) {
                listDevices = true;
            } else if(optArg.equals("", "--continue")) {
                _config.checkpointFile = optArg.arg;
            } else if(optArg.equals("", "--keyspace")) {
                secp256k1::uint256 start;
                secp256k1::uint256 end;

                parseKeyspace(optArg.arg, start, end);

                if(start.cmp(secp256k1::N) > 0) {
                    throw std::string("argument is out of range");
                }
                if(start.isZero()) {
                    throw std::string("argument is out of range");
                }

                if(end.cmp(secp256k1::N) > 0) {
                    throw std::string("argument is out of range");
                }

                if(start.cmp(end) > 0) {
                    throw std::string("Invalid argument");
                }

                _config.startKey = start;
                _config.nextKey = start;
                _config.endKey = end;
            } else if(optArg.equals("", "--share")) {
                if(!parseShare(optArg.arg, shareIdx, numShares)) {
                    throw std::string("Invalid argument");
                }
                optShares = true;
            } else if(optArg.equals("", "--stride")) {
                try {
                    _config.stride = secp256k1::uint256(optArg.arg);
                } catch(...) {
                    throw std::string("invalid argument: : expected hex string");
                }

                if(_config.stride.cmp(secp256k1::N) >= 0) {
                    throw std::string("argument is out of range");
                }

                if(_config.stride.cmp(0) == 0) {
                    throw std::string("argument is out of range");
                }
            } else if(optArg.equals("", "--create-ranges")) {
                rangesCreateFile = optArg.arg;
                optCreateRanges = true;
            } else if(optArg.equals("", "--process-ranges")) {
                rangesProcessFile = optArg.arg;
                optProcessRanges = true;
            } else if(optArg.equals("-f", "--follow")) {
                _config.follow = true;
            }

		} catch(std::string err) {
			Logger::log(LogLevel::Error, "Error " + opt + ": " + err);
			return 1;
		}
	}

    if(listDevices) {
        printDeviceList(_devices);
        return 0;
    }

	// Verify device exists
	if(_config.device < 0 || _config.device >= _devices.size()) {
		Logger::log(LogLevel::Error, "device " + util::format(_config.device) + " does not exist");
		return 1;
	}

	// Parse operands
	std::vector<std::string> ops = parser.getOperands();

    // If there are no operands, then we must be reading from a file, otherwise
    // expect addresses on the commandline
	if(ops.size() == 0) {
		if(_config.targetsFile.length() == 0) {
			Logger::log(LogLevel::Error, "Missing arguments");
			usage();
			return 1;
		}
	} else {
		for(unsigned int i = 0; i < ops.size(); i++) {
            if(!Address::verifyAddress(ops[i])) {
                Logger::log(LogLevel::Error, "Invalid address '" + ops[i] + "'");
                return 1;
            }
			_config.targets.push_back(ops[i]);
		}
	}
    
    // Calculate where to start and end in the keyspace when the --share option is used
    if(optShares) {
        Logger::log(LogLevel::Info, "Share " + util::format(shareIdx) + " of " + util::format(numShares));
        secp256k1::uint256 numKeys = _config.endKey - _config.nextKey + 1;

        secp256k1::uint256 diff = numKeys.mod(numShares);
        numKeys = numKeys - diff;

        secp256k1::uint256 shareSize = numKeys.div(numShares);

        secp256k1::uint256 startPos = _config.nextKey + (shareSize * (shareIdx - 1));

        if(shareIdx < numShares) {
            secp256k1::uint256 endPos = _config.nextKey + (shareSize * (shareIdx)) - 1;
            _config.endKey = endPos;
        }

        _config.nextKey = startPos;
        _config.startKey = startPos;
    }

	// Check option for compressed, uncompressed, or both
	if(optCompressed && optUncompressed) {
		_config.compression = PointCompressionType::BOTH;
	} else if(optCompressed) {
		_config.compression = PointCompressionType::COMPRESSED;
	} else if(optUncompressed) {
		_config.compression = PointCompressionType::UNCOMPRESSED;
	}

    if(_config.checkpointFile.length() > 0) {
        readCheckpointFile();
    }

    if(optCreateRanges) {
        createRangesFile(rangesCreateFile);
        return 0;
    }

    if(optProcessRanges) {
        return processRanges(rangesProcessFile);
    }

    return run();
}