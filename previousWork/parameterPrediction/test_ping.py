import ping3

TargetIP = '1.116.234.93'

n = 5

TotalTime = 0.0
for k in range(n):
    IPstrip = TargetIP.strip()
    PingTime = ping3.ping(IPstrip, timeout=1, unit='ms')
    print(str(k) + '  ' + str(PingTime) + 'ms')
    if not PingTime:
        PingTime = 5000.0
    TotalTime = TotalTime+PingTime
print(IPstrip + '   average ping:' + str((TotalTime/n)) + 'ms')


