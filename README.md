## Digit Recognizer with Keras
This project builds a digit recognizer based on deep neural networks. `Keras` is used to construct the system. 

The dataset and the problem is from [Kaggle](https://www.kaggle.com/c/digit-recognizer).

## How to configure and run the system
You need to download the folders [data](https://github.com/JupiterEthan/Digit-Recognizer-with-Keras/tree/master/data) and [scripts](https://github.com/JupiterEthan/Digit-Recognizer-with-Keras/tree/master/scripts). When you do that, please make sure the original directory tree is kept.

Now you should be able to find the main file `DigitRecognizerDNN.py` in `scripts`. You can set up or modify the system configurations in this file.

When you run `DigitRecognizerDNN.py` with `python` command in the terminal, you will obtain the results in the `results` folder.

In my experiment, I got a public score of `0.96000`. Since this project just simply feeds the DNN with the vector representation of the images, the 2-dimensional structure information of the images have been obviously missing. So if CNN is used, the score should be even higher. The CNN solution will be given soon.

