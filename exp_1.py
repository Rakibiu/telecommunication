import numpy as np
import matplotlib.pyplot as plt

# Parameters
fm = 5
fs = 50
bits = 3

# 1. Original signal
t = np.linspace(0, 1, 500)
x = np.sin(2*np.pi*fm*t)

# 2. Sampling
ts = np.arange(0, 1, 1/fs)
sample = np.sin(2*np.pi*fm*ts)

# 3. Quantization
levels = 2**bits
q = np.round((sample + 1) * (levels-1) / 2)

# 4. PCM Encoding
pcm = [format(int(i), '03b') for i in q]

print("PCM Codes:")
print(pcm)
print("PCM Codes as Bits:",ts)

# 5. Demodulation
demod = q * 2/(levels-1) - 1

# Graphs
plt.figure(figsize=(8,10))

plt.subplot(5,1,1)
plt.plot(t, x)
plt.title("Original Signal")
plt.grid()

plt.subplot(5,1,2)
plt.stem(ts, sample)
plt.title("Sampled Signal")
plt.grid()

plt.subplot(5,1,3)
plt.step(ts, q, where='mid')
plt.title("Quantized Signal")
plt.grid()

# Convert PCM codes to bits
bits_data = [int(b) for code in pcm for b in code]

plt.subplot(5,1,4)
plt.step(range(len(bits_data)), bits_data)
plt.title("PCM Encoded Signal")
plt.ylim(-0.2, 1.2)
plt.grid()

plt.subplot(5,1,5)
plt.plot(t, x, label="Original")
plt.plot(ts, demod, 'o-', label="Demodulated")
plt.title("Demodulated Signal")
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()
