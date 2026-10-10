import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)

# ==========================================================
# 1. PARAMETERS + RANDOM DATA
# ==========================================================

N = 5000
SNR = np.arange(0, 11, 2)

data = np.random.randint(0, 2, N)


# ==========================================================
# 2. (2,1,2) CONVOLUTIONAL ENCODER
# Generator: g1 = 111, g2 = 101
# ==========================================================

def encoder(data):
    m1 = m2 = 0
    code = []

    for b in data:
        code += [b ^ m1 ^ m2, b ^ m2]
        m2, m1 = m1, b

    return np.array(code)


# Add 2 zeros to clear memory
encoded = encoder(np.append(data, [0, 0]))


# ==========================================================
# 3. BPSK MODULATION
# 0 -> +1, 1 -> -1
# ==========================================================

bpsk = 1 - 2 * encoded


# ==========================================================
# 4. VITERBI DECODER
# ==========================================================

def viterbi(r):

    n = len(r) // 2

    metric = np.array([0, 999999, 999999, 999999])
    prev = np.zeros((n, 4), int)
    bits = np.zeros((n, 4), int)

    for i in range(n):

        new = np.ones(4) * 999999

        for state in range(4):

            m1, m2 = state // 2, state % 2

            for b in [0, 1]:

                c1 = b ^ m1 ^ m2
                c2 = b ^ m2

                next_state = 2*b + m1

                d = (c1 != r[2*i]) + (c2 != r[2*i+1])
                value = metric[state] + d

                if value < new[next_state]:
                    new[next_state] = value
                    prev[i, next_state] = state
                    bits[i, next_state] = b

        metric = new

    # Traceback
    state = 0
    decoded = []

    for i in range(n-1, -1, -1):
        decoded.append(bits[i, state])
        state = prev[i, state]

    return np.array(decoded[::-1])


# ==========================================================
# 5. AWGN CHANNEL + DEMODULATION + BER
# ==========================================================

BER = []
errors = []

for snr in SNR:

    # SNR dB -> linear
    snr_linear = 10**(snr/10)

    # Noise variance
    noise_var = 1 / snr_linear

    # AWGN
    noise = np.sqrt(noise_var) * np.random.randn(len(bpsk))

    # Channel
    received = bpsk + noise

    # 6. BPSK Demodulation
    demod = (received < 0).astype(int)

    # 7. Viterbi decoding
    decoded = viterbi(demod)[:N]

    # 8. Error / mismatch
    error = np.sum(data != decoded)

    # 9. BER
    ber = error / N

    errors.append(error)
    BER.append(ber)

    print(f"SNR={snr:2d} dB   Errors={error:4d}   BER={ber:.6f}")


# ==========================================================
# Example signal at 4 dB for plotting
# ==========================================================

snr = 4
noise_var = 1 / (10**(snr/10))

noise = np.sqrt(noise_var) * np.random.randn(len(bpsk))
received = bpsk + noise

demod = (received < 0).astype(int)
decoded = viterbi(demod)[:N]


# ==========================================================
# 10. SMOOTH SIGNALS FOR DISPLAY
# ==========================================================

show = 8
samples = 100
fc = 5

# Input waveform
input_wave = np.repeat(data[:show], samples)

# Encoded waveform
enc_wave = np.repeat(encoded[:2*show], samples)

# Time for BPSK
t = np.arange(2 * show * samples) / samples

# Smooth carrier
carrier = np.cos(2*np.pi*fc*t)

# Smooth BPSK
symbol_wave = np.repeat(bpsk[:2*show], samples)
bpsk_wave = symbol_wave * carrier

# Smooth AWGN received waveform
noise_wave = np.sqrt(noise_var/2) * np.random.randn(len(bpsk_wave))
received_wave = bpsk_wave + noise_wave

# Decoded waveform
decoded_wave = np.repeat(decoded[:show], samples)


# ==========================================================
# 11. INPUT, ENCODED, BPSK, RECEIVED, DECODED GRAPHS
# ==========================================================

plt.figure(figsize=(10, 10))

plt.subplot(5,1,1)
plt.plot(input_wave)
plt.title("1. Input Binary Data")
plt.ylim(-0.2, 1.2)
plt.grid()

plt.subplot(5,1,2)
plt.plot(enc_wave)
plt.title("2. Convolutionally Encoded Data")
plt.ylim(-0.2, 1.2)
plt.grid()

plt.subplot(5,1,3)
plt.plot(bpsk_wave)
plt.title("3. BPSK Modulated Signal")
plt.grid()

plt.subplot(5,1,4)
plt.plot(received_wave)
plt.title("4. Received Signal after AWGN")
plt.grid()

plt.subplot(5,1,5)
plt.plot(decoded_wave)
plt.title("5. Viterbi Decoded Data")
plt.xlabel("Samples")
plt.ylim(-0.2, 1.2)
plt.grid()

plt.tight_layout()
plt.show()


# ==========================================================
# 12. BER vs SNR
# ==========================================================

plt.figure()

# semilogy cannot show BER = 0
BER_plot = np.maximum(BER, 1/N)

plt.semilogy(SNR, BER_plot, 'o-')

plt.title("BER vs SNR")
plt.xlabel("SNR (dB)")
plt.ylabel("BER")
plt.grid(True, which="both")

plt.show()