# Your test pictures

Put here your own pictures (PNG, JPG, any size and shape) and run

    python3 1_test_set/make_test_set.py

They become a second test set, `TEST_SET/mine`, scored apart from the
reference one. Choose the kind of pictures you want the denoiser to be good
at (for example QSL cards, text, drawings).

Never use the same pictures for training: a picture the network has already
seen gives a score that is too good. `2_training_data/make_training_data.py`
checks this and leaves out of the training data any picture that is also
here.
