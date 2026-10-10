import numpy as np
import matplotlib.pyplot as plt

# 1. Input data
data = np.array([1, 0, 1, 1])

# 2. 7-chip PN code
pn = np.array([1, 0, 1, 1, 0, 1, 0])
pn = 2 * pn - 1          # 0 -> -1, 1 -> +1

# 3. BPSK mapping
bpsk = 2 * data - 1      # 1 -> +1, 0 -> -1

# 4. DSSS spreading
spread = np.concatenate([bit * pn for bit in bpsk])

print("Input Data :", data)
print("PN Code    :", pn)
print("BPSK Data  :", bpsk)
print("Total Chips:", len(spread))


# 5. Create smooth signal
samples = 100
spread_wave = np.repeat(spread, samples)

# 6. Time
t = np.linspace(0, 5, len(spread_wave))

# 7. Low-frequency carrier
fc = 10
carrier = np.cos(2 * np.pi * fc * t)

# 8. Final DSSS-BPSK signal
final_signal = spread_wave * carrier

# 9. Input data waveform
data_wave = np.repeat(data, 7 * samples)


# =========================
# 10. Plot
# =========================

plt.figure(figsize=(14, 10))

# Input data
plt.subplot(4, 1, 1)
plt.plot(t, data_wave)
plt.title("Input Binary Data")
plt.ylabel("Amplitude")
plt.ylim(-0.2, 1.2)
plt.grid()

# DSSS spread signal
plt.subplot(4, 1, 2)
plt.plot(t, spread_wave)
plt.title("DSSS Spread Signal (7 Chips per Bit)")
plt.ylabel("Amplitude")
plt.ylim(-1.2, 1.2)
plt.grid()

# Carrier
plt.subplot(4, 1, 3)
plt.plot(t, carrier)
plt.title("BPSK Carrier Signal")
plt.ylabel("Amplitude")
plt.ylim(-1.2, 1.2)
plt.grid()

# Final DSSS-BPSK
plt.subplot(4, 1, 4)
plt.plot(t, final_signal)
plt.title("Final DSSS-BPSK Signal")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.ylim(-1.2, 1.2)
plt.grid()

plt.tight_layout()
plt.show()