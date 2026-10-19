# Step 4 – Check and score

Two questions for a trained model:

1. does the C code give **the same pictures** as PyTorch?
2. **how much does it clean**, compared with the reference model and the
   classic denoiser?

## 4a – The C code matches PyTorch

    python3 4_check_and_score/check_c_matches_pytorch.py TRAINED/20x6

The folder is the output of `train.py`. The script cleans `check_in.png`
with the C code (`bin/sstv_denoise_tool`, the code of the radio) and
compares it with `check_torch.png`, made by PyTorch from the same picture.
Then it compares the fast vector loops of the C code with its plain loops.

- **`OK`**: at most 2 levels of difference (out of 255), from rounding. The
  model works in C exactly as it was trained.
- **`DIFFERENT`**: the model file and the C code do not agree. Do not use
  the model.

## 4b – The score

    python3 4_check_and_score/score.py TRAINED/20x6/sstv_dnn_20x6.bin models/sstv_dnn_20x6.bin

Any number of models. Each one cleans every noisy picture of the test sets
with the C code, and is compared with the clean picture. The classic
(non-ML) denoiser of the radio is always added, as a reference.

    === test set reference (TEST_SET/reference)
    noise                n  received     classic        20x6        16x7
    ALL                158    19.57       +5.24       +9.15       +8.66
    ALL detail         158    19.32       +2.41       +4.52       ...
    snr0                11    12.70       +5.90      +11.64      +11.49
    snr3                21    15.72       +6.00      +10.53      +10.31
    ...
    burst_snr6          21    18.76       +5.70       +9.62       +9.21
    ...

| Column | What |
|---|---|
| `noise` | the noise case (see `1_test_set/README.md`); **`ALL`** is the mean over every noisy picture: **the number to compare**. **`ALL detail`** is the same mean measured only on the **detail** of each picture: the pixels next to a strong brightness step in the clean picture – letters, thin lines, contours (`detail_mask` in `common/sstvml.py`). A denoiser that softens text loses much more here than in `ALL` |
| `n` | how many pictures |
| `received` | PSNR of the received picture against the clean one, in dB: how close it is (higher = closer) |
| `classic`, models | **how many dB each denoiser adds** to the received picture |

Then, for each model, how much it changes the **clean** pictures: the
PSNR between a clean picture and the same picture after the model. Higher
is better; above about 40 dB the change cannot be seen. A good denoiser
leaves a clean picture alone. (On the radio a clean picture is not even
given to the network, see step 5, but a model that damages clean pictures
also damages the clean parts of noisy ones.)

Each test set (`reference`, and `mine` if you made it) has its own table:
never mix them. Everything is also written to `RESULTS/scores/scores.csv`.

| Option | What |
|---|---|
| `--jobs N` | pictures cleaned at the same time (default: all the cores) |
| `--keep` | keeps the cleaned pictures in `RESULTS/scores/<set>/<model>/` |

## What PSNR means

PSNR (peak signal-to-noise ratio) compares two pictures pixel by pixel:

    PSNR = 10 · log10(255² / mean squared difference)      in dB

+3 dB means the remaining error has half the power; +10 dB, a tenth. It is
a good measure for comparing denoisers on the same pictures, but not the
whole story: two results with the same PSNR can look different (one
softer, one with more small speckles). Look at the pictures too: step 6
shows them side by side.

## The tool

`bin/sstv_denoise_tool` can also be used by hand:

    bin/sstv_denoise_tool ml        MODEL.bin IN.png OUT.png     the network, always
    bin/sstv_denoise_tool radio     MODEL.bin IN.png OUT.png     as the radio (noise gate)
    bin/sstv_denoise_tool classic   IN.png OUT.png               the classic denoiser
    bin/sstv_denoise_tool measure   IN.png                       only the noise measure
    bin/sstv_denoise_tool selfcheck MODEL.bin IN.png             fast loops = plain loops?
