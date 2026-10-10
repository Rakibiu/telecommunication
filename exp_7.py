import numpy as np
import matplotlib.pyplot as plt

# 1. Input data
data = np.array([1, 0, 1, 1, 0, 1, 0])

# 2. Seven hopping frequencies
freq = [1, 2, 3, 4, 5, 6, 7]

# 3. Random frequency hopping pattern
hop = np.random.choice(freq, len(data))

# 4. Time for one bit
N = 100
t = np.arange(N) / N

# Store signals
original = []
bpsk = []
spread = []
fhss = []

# 5. Generate signals
for i in range(len(data)):

    # BPSK: 1 -> +1, 0 -> -1
    bit = 2 * data[i] - 1

    # Carrier
    carrier = np.cos(2 * np.pi *10* t)

    # BPSK signal
    bpsk_signal = bit * carrier

    # Hopping frequency
    f = hop[i]

    # Spread carrier
    hop_signal = np.cos(2 * np.pi * f * t)

    # Final FHSS signal
    final = bpsk_signal * hop_signal

    # Save signals
    original.extend([bit] * N)
    bpsk.extend(bpsk_signal)
    spread.extend(hop_signal)
    fhss.extend(final)

# 6. Print results
print("Input Data:", data)
print("Hopping Frequencies:", hop)

# 7. Plot
plt.subplot(4, 1, 1)
plt.plot(original)
plt.title("Original Bit Sequence")
plt.grid()

plt.subplot(4, 1, 2)
plt.plot(bpsk)
plt.title("BPSK Modulated Signal")
plt.grid()

plt.subplot(4, 1, 3)
plt.plot(spread)
plt.title("Spread Signal with 7 Frequencies")
plt.grid()

plt.subplot(4, 1, 4)
plt.plot(fhss)
plt.title("Frequency Hopped Spread Spectrum Signal")
plt.xlabel("Time")
plt.grid()

plt.tight_layout()
plt.show()