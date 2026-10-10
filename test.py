import numpy as np
import matplotlib.pyplot as plt
data=np.array([1,0,1,0,1,1,0])
freq=[1,2,3,4,5,6,7]
hop=np.random.choice(freq,len(data))
N=100
t=np.arange(N)/N
orginal=[]
bpsk=[]
spread=[]
fhss=[]
for i in range(len(data)):
    bit=2*data[i]-1
    carrier=np.cos(2*np.pi*10*t)
    bpsk_signal=bit*carrier
    f=hop[i]
    spread_signal=np.cos(2*np.pi*f*t)
    fhss_signal=bpsk_signal*spread_signal

    orginal.extend([bit]*N)
    bpsk.extend(bpsk_signal)
    spread.extend(spread_signal)
    fhss.extend(fhss_signal)

print("Input Data:", data)
print("Hopping Frequencies:", hop)



plt.subplot(4,1,1)
plt.plot(orginal)
plt.title("Original Bit Sequence")
plt.grid()
plt.subplot(4,1,2)
plt.plot(fhss)
plt.title("Original Bit Sequence")
plt.grid()

plt.show()

