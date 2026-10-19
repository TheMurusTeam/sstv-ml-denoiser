# Step 2 – The training data

The network learns from **pairs** of pictures: the same picture received
with noise and without noise. This step makes the pairs from **your
pictures**, with the real SSTV encoder and decoder of the radio, so no real
received picture is needed: only many pictures.

    python3 2_training_data/make_training_data.py --pictures FOLDER [FOLDER ...]

Several folders can be given, for example your photos, your QSL cards and
the text pictures made by `make_text_pictures.py` (below):

    python3 2_training_data/make_training_data.py --pictures ~/photos ~/qsl PICTURES_TEXT

Do **step 1 first**: this step checks that no training picture is also a
test picture.

## Which pictures

- Any format (PNG, JPG…), any size, any shape, in one or more folders
  (subfolders are not read; files starting with `_`, like `_check.jpg`,
  are skipped). They are prepared here, as in step 1.
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
| `--pictures` | – | one or more folders of pictures |
| `--out` | `DATA` | output folder |
| `--modes-per-picture` | 2 | SSTV modes per picture |
| `--variants` | 5 | noisy receptions per picture and mode |
| `--threads` | all the cores | simulations at the same time |
| `--seed` | 7 | the random choices: same pictures and same seed give the same data |
| `--no-test-check` | off | skips the check against the test sets (not recommended) |

## Pictures full of text – `make_text_pictures.py`

Photos have little text, so a network trained on photos sees few letters and
learns to soften them. SSTV pictures – QSL cards above all – are full of
callsigns and reports. This script makes pictures like that, to add to your
own:

    python3 2_training_data/make_text_pictures.py --count 500 --backgrounds ~/photos

Each picture (640×512, brought to 320×256 by step 2, so the letters are
smooth as in a real picture) has:

- a background: a piece of one of your photos (`--backgrounds`), or a colour
  gradient, a flat colour, soft colour blobs, or black. Background photos
  that are test pictures of step 1 are left out: hidden behind some text,
  step 2 could not recognise them any more;
- a big callsign (amateur like `IZ1ABC`, or CB like `1AT123`) and 2 to 6
  more pieces of text: QSO words, reports, frequencies, dates, locators,
  names – from big titles down to lines of 7–8 pixels;
- for each piece its own font, colour and style: plain, outlined, with a
  shadow, on a box, a little rotated; sometimes thin lines and frames, as on
  QSL cards.

The result is in `PICTURES_TEXT/`, with `_examples.jpg` to have a look. The
same `--seed` gives the same pictures.

| Option | Default | What |
|---|---|---|
| `--count` | 500 | how many pictures |
| `--backgrounds` | – | a folder of photos for the backgrounds (optional) |
| `--system-fonts` | off | also use the fonts installed on your computer, for more variety |
| `--out` | `PICTURES_TEXT` | output folder |
| `--seed` | 11 | the random choices |

**How many?** Text should be a good share of the training, not all of it:
about one text picture for every one or two photos is a reasonable start.
The network must still learn photos.

**Fonts:** the free fonts in `fonts/` (SIL Open Font License, see
`fonts/FONTS.md`), plus any `.ttf`/`.otf` you add there.

## Why the clean picture is a received picture

The target is not the original picture but the picture **received without
noise**. SSTV itself softens the picture a little (the FM signal carries a
limited detail along the line, Robot sends the colour at half resolution).
If the target were the original, the network would also try to "improve"
SSTV and would invent detail; with the received picture as target, it
learns to remove **only the noise**.
