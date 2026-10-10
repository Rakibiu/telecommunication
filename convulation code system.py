
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)  # Reproducible random results

# 1. Parameters and random data
N = 5000
SNR = np.arange(0, 11, 2)  # SNR: 0, 2, 4, 6, 8, 10 dB
data = np.random.randint(0, 2, N)  # Generate random bits


# 2. Convolutional encoder (rate 1/2, K=3)
def encoder(data):
    m1 = m2 = 0
    code = []

    for b in data:
        code += [b ^ m1 ^ m2, b ^ m2]  # Generate 2 coded bits
        m2, m1 = m1, b                # Update memory

    return np.array(code)

# Add two zero tail bits and encode
encoded = encoder(np.append(data, [0, 0]))


# 3. BPSK modulation: 0 -> +1, 1 -> -1
bpsk = 1 - 2 * encoded


# 4. Viterbi decoder
def viterbi(r):
    n = len(r) // 2  # Number of input steps

    metric = np.array([0, 999999, 999999, 999999])  # Initial path metrics
    prev = np.zeros((n, 4), int)  # Store previous states
    bits = np.zeros((n, 4), int)  # Store decoded input bits

    for i in range(n):
        new = np.ones(4) * 999999  # New path metrics

        for state in range(4):
            m1, m2 = state // 2, state % 2  # Extract memory bits

            for b in [0, 1]:  # Try both possible input bits
                c1 = b ^ m1 ^ m2
                c2 = b ^ m2                 # Expected coded bits
                next_state = 2*b + m1        # Calculate next state

                # Hamming distance between expected and received bits
                d = int(c1 != r[2*i]) + int(c2 != r[2*i+1])
                value = metric[state] + d   # Update path metric

                if value < new[next_state]: # Keep the best path
                    new[next_state] = value
                    prev[i, next_state] = state
                    bits[i, next_state] = b

        metric = new  # Update metrics

    # Trace back from final state 00
    state = 0
    decoded = []

    for i in range(n-1, -1, -1):
        decoded.append(bits[i, state])
        state = prev[i, state]

    return np.array(decoded[::-1])  # Restore bit order


# 5. AWGN channel, demodulation, and BER
BER = []
errors = []

for snr in SNR:
    snr_linear = 10**(snr/10)  # Convert dB to linear SNR
    noise_var = 1 / snr_linear  # Calculate noise variance

    noise = np.sqrt(noise_var) * np.random.randn(len(bpsk))
    received = bpsk + noise     # Add AWGN to BPSK symbols

    demod = (received < 0).astype(int)  # BPSK demodulation
    decoded = viterbi(demod)[:N]        # Decode and remove tail bits

    error = np.sum(data != decoded)     # Count bit errors
    ber = error / N                     # Calculate BER

    errors.append(error)
    BER.append(ber)

    print(f"SNR={snr:2d} dB   Errors={error:4d}   BER={ber:.6f}")


# 6. Generate an example received signal at 4 dB
snr = 4
noise_var = 1 / (10**(snr/10))

noise = np.sqrt(noise_var) * np.random.randn(len(bpsk))
received = bpsk + noise
demod = (received < 0).astype(int)
decoded = viterbi(demod)[:N]


# 7. Prepare signals for plotting
show = 8
samples = 100
fc = 5

input_wave = np.repeat(data[:show], samples)        # Input waveform
enc_wave = np.repeat(encoded[:2*show], samples)     # Encoded waveform

t = np.arange(2 * show * samples) / samples
carrier = np.cos(2*np.pi*fc*t)                     # Cosine carrier

symbol_wave = np.repeat(bpsk[:2*show], samples)
bpsk_wave = symbol_wave * carrier                  # BPSK waveform

noise_wave = np.sqrt(noise_var/2) * np.random.randn(len(bpsk_wave))
received_wave = bpsk_wave + noise_wave             # Noisy waveform

decoded_wave = np.repeat(decoded[:show], samples)  # Decoded waveform


# 8. Plot input, encoded, modulated, received, and decoded signals
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


# 9. Plot BER versus SNR
plt.figure()

BER_plot = np.maximum(BER, 1/N)  # Replace zero only for plotting

plt.semilogy(SNR, BER_plot, 'o-')
plt.title("BER vs SNR")
plt.xlabel("SNR (dB)")
plt.ylabel("BER")
plt.grid(True, which="both")
plt.show()
