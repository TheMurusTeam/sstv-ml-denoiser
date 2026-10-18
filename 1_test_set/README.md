# Step 1 – The test sets

A **test set** is a set of pictures a model is measured on and **never
trained on**. It tells how well a model cleans pictures it has never seen,
which is the only thing that matters.

    python3 1_test_set/make_test_set.py

Takes about half a minute for the reference pictures.

## Two test sets

| Source folder | Test set | What it is for |
|---|---|---|
| `1_test_set/reference_pictures/` | `TEST_SET/reference/` | the pictures that come with this package. They are the same for everybody, so scores can be compared with `models/README.md` and with anybody else's. Do not change them. |
| `1_test_set/my_pictures/` | `TEST_SET/mine/` | **your** pictures (optional): the kind of pictures you want the denoiser to be good at, for example QSL cards with text. Scored apart from the reference. |

The two are never mixed: a score on `mine` depends on your pictures and
cannot be compared with anybody else's.

Run the script again whenever you add or remove pictures in `my_pictures/`.

## What the script does

1. **Brings every picture to 320×256.** SSTV sends exactly 320×256 pixels
   (Martin and Scottie; Robot 36 and 72 send 320×240). The pictures in
   `reference_pictures/` have other sizes on purpose: like any picture of
   yours, they have to be prepared first. Each one is:
   - turned as the camera says (EXIF orientation);
   - **cropped at the centre** to the 5:4 shape of 320×256: the long side is
     cut, nothing is squeezed and no black bands are added;
   - **scaled down or up** to exactly 320×256.

   The result is in `TEST_SET/<set>/pictures_320/`: have a look, the crop
   is what will be sent. Pictures enlarged more than 2 times are reported:
   they come out soft, a bigger original is better.

2. **Receives every picture through SSTV**, with the encoder and the
   decoder of the radio (`bin/sstv_channel_sim`), in 3 modes: Martin M1,
   Scottie S1, Robot 36.
   - Once **without noise**: the **clean** picture, the target of the
     denoiser. It has the small defects of SSTV itself (soft edges, colours
     a little spread) but no noise.
   - Then with the **8 noise cases** below: the **noisy** pictures.

## The noise cases

| Case | What |
|---|---|
| `snr0`, `snr3`, `snr6`, `snr10` | white noise; the SNR is measured in 2500 Hz, as for SSB. 0 dB is a picture barely readable, 10 dB a good one |
| `burst_snr6`, `burst_snr12` | noise plus bursts of strong interference (0.3–1.2 s, about 3 a minute): the coloured bands across the picture |
| `impulses_snr10` | noise plus short clicks (2–32 ms, about 10 a minute) |
| `fading_snr10` | noise plus a slow fading (0.15–0.45 Hz) down to 10 dB below |

The noise is **fixed**: the seed of each reception comes from its file
name, so the same picture always gives the same noisy pictures, on every
computer. Adding a picture does not change the others.

A few receptions are **dropped** and reported, mostly at 0 dB: the decoder
lost the line sync and the picture came out shifted or cut. Such a picture
cannot be compared pixel by pixel with the clean one. This is normal.

## The folders it makes

    TEST_SET/reference/pictures_320/<name>.png                the prepared pictures
    TEST_SET/reference/clean/<name>_<mode>.png                received without noise
    TEST_SET/reference/noisy/<name>_<mode>__<case>.png        received with noise
    TEST_SET/mine/...                                         the same for my_pictures/

The part before `__` links each noisy picture to its clean one.
`TEST_SET` is made from scratch every time; it can be deleted.

## The reference pictures

Seven pictures, chosen to be different from each other: a portrait, an
animal, objects, a starry sky, a microscope picture, a rocket launch and a
**test card** with thin lines and small text, the hardest case for a
denoiser. Their credits and licenses are in
`reference_pictures/CREDITS.md`.
