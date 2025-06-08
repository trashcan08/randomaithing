# Linear neural network for recognising MNist handwritten digits
This project is a linear neural network for recognising handwritten digits from the MNist dataset. This was completely done in processing, and no neural network libraries were used. Yes, I did all the math, and yes, it was a nightmare to debug.

The project consists of:
- customisable settings for number of epochs and number of images per epoch
- customisable hidden layers of neurons (though using the preset {32, 32} is recommended)
- customisable learning rate (though preset 0.01 is also recommended)
- displays guess and correct answer for each number
- displays total number of images classified correctly and wrongly, as well as accuracy overall (from all the epochs)
- shows a progress bar per epoch
- shows end-layer activations as a bar chart
- shows cost per epoch when any key is pressed (not available on first epoch)
- after training, you can draw on the space where the handwritten digits were displayed. (click to draw and right-click to erase). The drawing code is a little buggy though. The AI would then guess the digit drawn in real time.

## The network
- The network is just a linear neural network, taking in the 28 by 28 grid as a 784 1D array. 
- It uses the non-linear activation function ReLU
- It uses cross-entropy loss to calculate loss
- It uses Softmax for the last layer.

## How to run
Just get processing and run it lol, don't forget to set the filePath to your own filePath to the project. If you still cannot open it, I'm working on a OpenProcessing project which you get (hopefully) run on the web.
