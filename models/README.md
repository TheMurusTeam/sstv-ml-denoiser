# The reference model

`sstv_dnn_20x6.bin` is the network of the **1KO125 firmware** for the Xiegu
X6100 (Denoise ML in the SSTV application). It is here as the **reference**:
a model you train is better if it beats this one on the reference test set.

| | |
|---|---|
| Network | 20 channels × 6 layers (15 583 weights, 62 444 bytes) |
| Work per 320×256 picture | 1.27 G multiply-adds (about 1.3 s on the radio, a fraction of a second on a computer) |
| Training | the reference recipe of `3_train` (200 000 steps), on photos and QSL cards |

## Scores on the reference test set

Made with

    python3 4_check_and_score/score.py models/sstv_dnn_20x6.bin

on `TEST_SET/reference` (7 pictures × 3 modes, 158 noisy receptions). dB
added to the received picture:

| Noise | Pictures | Received (dB) | Classic | **20x6** |
|---|---|---|---|---|
| **ALL** | 158 | 19.57 | +5.24 | **+9.15** |
| **ALL detail** (text, lines, edges only) | 158 | 19.32 | +2.41 | **+4.52** |
| snr0 | 11 | 12.70 | +5.90 | +11.64 |
| snr3 | 21 | 15.72 | +6.00 | +10.53 |
| snr6 | 21 | 19.65 | +5.46 | +9.36 |
| snr10 | 21 | 23.75 | +4.26 | +7.50 |
| burst_snr6 | 21 | 18.76 | +5.70 | +9.62 |
| burst_snr12 | 21 | 23.23 | +4.83 | +7.79 |
| impulses_snr10 | 21 | 23.30 | +4.52 | +7.70 |
| fading_snr10 | 21 | 16.17 | +5.56 | +10.26 |

Clean pictures left alone: mean 44.8 dB, worst 26.1 dB (the test card).

On the detail of the pictures the gain is about half of the gain on the
whole pictures: the network removes the noise of smooth areas much better
than it keeps letters and thin lines sharp. That is where a better model has
the most to win (see "Text and fine detail" in `3_train/README.md`).

"Classic" is the classic (non-ML) denoiser of the same firmware: a measured,
edge-preserving filter with line and band repair. The network adds about
4 dB more.

**If the reference pictures change** (a new version of this package), the
numbers change too: always compare models on the same test set, scored the
same day with the same `score.py`.

## What the size gives

Other networks, trained by the author in the same way, on the same test set:

| Network | Multiply-adds | ALL | ALL detail |
|---|---|---|---|
| 16 × 7 | 1.01 G | +8.66 | +3.66 |
| **20 × 6** | 1.27 G | **+9.15** | **+4.52** |
| 20 × 7 | 1.56 G | +9.01 | +4.14 |
| 24 × 7 (long training) | 2.23 G | +9.82 | +4.64 |
| 32 × 8 | 4.67 G | +10.15 | +5.24 |

Bigger is usually a little better, but not always: the 20 × 7 is below the
20 × 6, and the 24 × 7 owes part of its gain to a longer training. The
training (pictures, steps) matters as much as the size. These models are not
included: train your own with `3_train`.

## The file format

`SSTVDNN1`, a format of this package: a header and the float numbers of
each layer, read by `src/sstv/sstv_dnn.c`. It is described in
`common/model_file.py`.
