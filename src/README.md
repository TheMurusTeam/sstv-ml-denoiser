# src – the C code

`make` at the top of the package builds the two tools into `bin/`.

| File | What |
|---|---|
| `sstv/encoder.c/.h` | SSTV encoder: a PNG becomes the audio of an SSTV transmission (Martin, Scottie, Robot) |
| `sstv/decoder.c/.h` | SSTV decoder: the audio becomes the received picture, with line sync and slant correction |
| `sstv/vis.c/.h`, `sstv/linesync.c/.h` | VIS code detector and line sync detector, used by the decoder |
| `sstv/sstv_denoise.c/.h` | the classic denoiser and the noise measure (the gate that decides whether the network runs) |
| `sstv/sstv_dnn.c/.h` | the neural network: reads the model file and cleans a picture, on several threads, NEON on ARM |
| `sstv_channel_sim.c` | **tool 1**: picture → SSTV audio → noise, impulses, bursts, fading → decoder → received picture |
| `sstv_denoise_tool.c` | **tool 2**: the denoisers on a picture, as the radio uses them (`ml`, `radio`, `classic`, `measure`, `selfcheck`) |
| `stubs/png.h`, `stubs/sndfile.h` | stand-ins for libpng and libsndfile, so nothing has to be installed |
| `third_party/stb_image*.h` | PNG/JPG reading and writing (public domain, github.com/nothings/stb) |

The files in `sstv/` are **the same files as in the 1KO125 firmware**
(`src/sstv/` of its source tree), not modified: this is what makes the
pictures here the same as on the radio. When the firmware changes them,
copy them here again.

Each tool tells its usage when run without arguments; their comments at the
top of the file explain every option.
