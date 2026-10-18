# Step 5 – Real received pictures

The test sets are simulated receptions. Here a model works on **real SSTV
pictures you received**, exactly as the radio would use it, and the result
is shown in an HTML page.

    python3 5_real_pictures/denoise_real_pictures.py --pictures FOLDER
    python3 5_real_pictures/denoise_real_pictures.py --pictures FOLDER --model TRAINED/16x7/sstv_dnn_16x7.bin --classic

`FOLDER` holds received SSTV pictures (PNG or JPG), as saved by the radio
or by any SSTV program. They are **not resized**: they are already SSTV
pictures. Files whose name ends in `_dn` (copies already cleaned by the
radio) are skipped.

## What happens to each picture

The same as on the radio at the end of a reception with Denoise ML:

1. the **noise is measured** (the measure of the classic denoiser, on the
   picture itself);
2. a **clean picture** (noise under the gate) is **left as it is**: the
   network does not even run. A good reception is never touched;
3. otherwise the **network cleans** the received picture. Black rows at the
   bottom (a picture not complete) are left out.

With `--always` the network runs on every picture, clean ones too: useful to
see whether a model damages good pictures.

## The page

`RESULTS/real_pictures/<model>/index.html` opens in the browser at the end:

- one row per picture: **received | ML** (and **classic** with `--classic`);
- above the pictures: size, measured noise, and whether the network ran;
- the **zoom** buttons enlarge every picture, pixels kept sharp;
- **press and hold** a cleaned picture to see the received one in its
  place: what changes is what the denoiser removed.

The folder also holds `pictures/` (the pictures of the page) and
`results.csv` (noise, network ran, multiply-adds). The whole folder can be
zipped or moved: the page uses only relative links.

## Options

| Option | Default | What |
|---|---|---|
| `--pictures` | – | the folder of received pictures |
| `--model` | `models/sstv_dnn_20x6.bin` | the model |
| `--classic` | off | adds the classic denoiser |
| `--always` | off | runs the network on clean pictures too |
| `--out` | `RESULTS/real_pictures/<model>` | output folder |
| `--jobs` | all the cores | pictures at the same time |
| `--no-browser` | off | does not open the page |

To compare several models on the same pictures, use step 6.
