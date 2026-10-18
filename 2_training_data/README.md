# Step 2 – The training data

The network learns from **pairs** of pictures: the same picture received
with noise and without noise. This step makes the pairs from **your
pictures**, with the real SSTV encoder and decoder of the radio, so no real
received picture is needed: only many pictures.

    python3 2_training_data/make_training_data.py --pictures FOLDER_OF_PICTURES

Do **step 1 first**: this step checks that no training picture is also a
test picture.

## Which pictures

- Any format (PNG, JPG…), any size, any shape, all in one folder
  (subfolders are not read). They are prepared here, as in step 1.
- **A few hundred at least.** More, and more varied, is better: landscapes,
  people, text, drawings and graphics, dark and bright pictures, skies.
  The network learns what pictures look like; whatever is missing (for
  example text) it will clean less well.
- Pictures of at least 320×256 are best: smaller ones are enlarged and come
  out soft (they are reported).
- Use pictures you are allowed to use. Free sets of photos exist, for
  example the Unsplash Lite dataset (https://unsplash.com/data).

## What the script does

1. **Brings every picture to 320×256**: cropped at the centre to the 5:4
   shape, then scaled down or up (nothing squeezed, no black bands), as in
   step 1. Result in `DATA/pictures_320/`.

2. **Leaves out the pictures that are also in the test sets** (`TEST_SET/
   reference` and `TEST_SET/mine`), even resized, cropped a little or
   recompressed: they are found by a fingerprint of the picture. A model
   trained on its test pictures gets scores that are too good. Each one
   left out is reported.

3. **Receives every picture through SSTV** (`bin/sstv_channel_sim`) in 2
   modes picked at random among Martin M1/M2, Scottie S1/S2, Robot 36/72:
   - once **without noise**: the **clean** picture, the target;
   - then **5 times with noise**: SNR picked among 0…15 dB, and about 1 in 4
     receptions with bursts, 1 in 5 with impulses, 1 in 7 with fading.

   A noisy reception where the decoder lost the line sync (picture shifted
   or cut) is **dropped**: a pair that is not aligned pixel by pixel teaches
   the network to blur.

## The folder it makes

    DATA/pictures_320/<name>.png                     the prepared pictures
    DATA/clean/<name>_<mode>.png                     received without noise
    DATA/noisy/<name>_<mode>__v<k>_snr<dB>.png       received with noise

The part before `__` links each noisy picture to its clean one: this is the
layout that `3_train/train.py` reads. `DATA` is made from scratch every
time; it can be deleted.

300 pictures take a few minutes and make about 2800 pairs (about 750 MB).

## Options

| Option | Default | What |
|---|---|---|
| `--pictures` | – | the folder of your pictures |
| `--out` | `DATA` | output folder |
| `--modes-per-picture` | 2 | SSTV modes per picture |
| `--variants` | 5 | noisy receptions per picture and mode |
| `--threads` | all the cores | simulations at the same time |
| `--seed` | 7 | the random choices: same pictures and same seed give the same data |
| `--no-test-check` | off | skips the check against the test sets (not recommended) |

## Why the clean picture is a received picture

The target is not the original picture but the picture **received without
noise**. SSTV itself softens the picture a little (the FM signal carries a
limited detail along the line, Robot sends the colour at half resolution).
If the target were the original, the network would also try to "improve"
SSTV and would invent detail; with the received picture as target, it
learns to remove **only the noise**.
