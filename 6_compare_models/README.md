# Step 6 – Compare models

One HTML page with several models side by side: their scores and their
pictures.

    python3 6_compare_models/compare_models.py MODEL_A.bin MODEL_B.bin [...] [--pictures FOLDER]

    python3 6_compare_models/compare_models.py models/sstv_dnn_20x6.bin \
            TRAINED/20x6/sstv_dnn_20x6.bin TRAINED/16x7/sstv_dnn_16x7.bin \
            --pictures ~/received_sstv

In the page the models are called by their file name without `sstv_dnn_`
(`20x6`, `16x7`…). Two models with the same file name get the name of their
folder too, for example `20x6 (models)` and `20x6 (20x6)`.

## The page

`RESULTS/compare_models/index.html` opens in the browser at the end:

1. **Scores** on each test set, as step 4 prints them, one column per
   model plus the classic denoiser, with the `ALL detail` row (the gain on
   text, thin lines and edges only). The best of each row is in green.
2. **One reception per test picture**, the same for every model (by
   default `snr3` in Martin M1, noisy enough to show the differences):

       received | clean (the target) | classic | model A | model B ...

   with the dB gain of each model above the pictures.
3. With `--pictures`, **your real received pictures**, each treated as the
   radio does (a clean picture is left as it is):

       received | model A | model B ...

As in step 5: the **zoom** buttons enlarge the pictures, and **pressing and
holding** a cleaned picture shows the received one in its place.

The scores say which model is better on average; the pictures say *how*:
softer or sharper, how well text survives, whether bursts are repaired.
Look at both.

## Options

| Option | Default | What |
|---|---|---|
| `--pictures` | – | real received pictures to add (see step 5) |
| `--example` | `snr3` | the test reception shown in part 2 (any case of `1_test_set/README.md`) |
| `--out` | `RESULTS/compare_models` | output folder |
| `--jobs` | all the cores | pictures at the same time |
| `--no-browser` | off | does not open the page |
