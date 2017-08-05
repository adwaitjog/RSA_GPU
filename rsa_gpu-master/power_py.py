#!/usr/bin/python
from subprocess import Popen, PIPE
from time import sleep
import numpy
import ivi
from sys import argv

# setup mso
mso = ivi.agilent.agilentMSOX4104A("TCPIP0::129.10.61.226::INSTR")
#mso.measurement.initiate()

if len(argv) >= 2:
	msg_num = int(argv[1])
else:
	msg_num = 112
if len(argv) >= 3:
	trace_num = int(argv[2])
else:
	trace_num = 10

power_trace = Popen(["./power_trace_local", str(msg_num), str(trace_num)], 
	stdout=PIPE, stdin=PIPE)
# print the first 4 lines
for i in range(4):
	print power_trace.stdout.readline(),

i = 0
power_v = numpy.zeros(trace_num);
file_name = "power_data_"+str(msg_num)+"_"+str(trace_num)+".bin"
while (power_trace.poll() == None):
	mso.measurement.initiate()
	if i==0:
		sleep(0.5)
	else:
		sleep(0.5)
	power_trace.stdin.write("a\n")
	print power_trace.stdout.readline(),
	wave = mso.channels[0].measurement.fetch_waveform()
	wave_int = [ord(x) for x in wave]
	power_mean = numpy.mean(wave_int)
	print power_mean
	power_v[i] = power_mean
#	file_name = "wave" + str(i) + ".bin"
#	with open(file_name, 'wb') as f:
#		f.write(wave)
	i += 1
	if i%1000 == 0:
		with open(file_name, "wb") as f:
			f.write(power_v)

print "EXIT"
print power_trace.stdout.readline(),
