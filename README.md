# Tractability of the Information Bottleneck for Deep Networks

Tishby and Zaslavky [2015] proposed a framework to explain the learning process of neural networks. 

They consider supervised learning, where the network has inputs <img src="https://rawgit.com/mostafarohani/info_bottle/master/svgs/cbfb1b2a33b28eab8a3e59464768e810.svg?invert_in_darkmode" align=middle width=14.852970000000001pt height=22.381919999999983pt/> and produces activations <img src="https://rawgit.com/mostafarohani/info_bottle/master/svgs/2f118ee06d05f3c2d98361d9c30e38ce.svg?invert_in_darkmode" align=middle width=11.845020000000003pt height=22.381919999999983pt/> as it predicts labels <img src="https://rawgit.com/mostafarohani/info_bottle/master/svgs/91aac9730317276af725abd8cef04ca9.svg?invert_in_darkmode" align=middle width=13.147200000000002pt height=22.381919999999983pt/>.
Examining the mutual informations <img src="https://rawgit.com/mostafarohani/info_bottle/master/svgs/ba4ad7d078f669d5c2c61a2eaf423293.svg?invert_in_darkmode" align=middle width=55.226655pt height=24.56552999999997pt/> and <img src="https://rawgit.com/mostafarohani/info_bottle/master/svgs/ab4b96f1e6312937210a8da7f5fa0b8d.svg?invert_in_darkmode" align=middle width=53.52963pt height=24.56552999999997pt/>, they observed two phases during the training process. 
The first is a "fitting" phase, in which <img src="https://rawgit.com/mostafarohani/info_bottle/master/svgs/ba4ad7d078f669d5c2c61a2eaf423293.svg?invert_in_darkmode" align=middle width=55.226655pt height=24.56552999999997pt/> and <img src="https://rawgit.com/mostafarohani/info_bottle/master/svgs/ab4b96f1e6312937210a8da7f5fa0b8d.svg?invert_in_darkmode" align=middle width=53.52963pt height=24.56552999999997pt/> jointly increase, as the learned features <img src="https://rawgit.com/mostafarohani/info_bottle/master/svgs/2f118ee06d05f3c2d98361d9c30e38ce.svg?invert_in_darkmode" align=middle width=11.845020000000003pt height=22.381919999999983pt/> go from randomly initialized to capturing information about both <img src="https://rawgit.com/mostafarohani/info_bottle/master/svgs/cbfb1b2a33b28eab8a3e59464768e810.svg?invert_in_darkmode" align=middle width=14.852970000000001pt height=22.381919999999983pt/> that helps predict <img src="https://rawgit.com/mostafarohani/info_bottle/master/svgs/91aac9730317276af725abd8cef04ca9.svg?invert_in_darkmode" align=middle width=13.147200000000002pt height=22.381919999999983pt/>.
The second is a "compression" phase, in which <img src="https://rawgit.com/mostafarohani/info_bottle/master/svgs/ab4b96f1e6312937210a8da7f5fa0b8d.svg?invert_in_darkmode" align=middle width=53.52963pt height=24.56552999999997pt/> continues to increase while <img src="https://rawgit.com/mostafarohani/info_bottle/master/svgs/ba4ad7d078f669d5c2c61a2eaf423293.svg?invert_in_darkmode" align=middle width=55.226655pt height=24.56552999999997pt/> decreases.
The interpretation of this is that the network tunes its representation of <img src="https://rawgit.com/mostafarohani/info_bottle/master/svgs/cbfb1b2a33b28eab8a3e59464768e810.svg?invert_in_darkmode" align=middle width=14.852970000000001pt height=22.381919999999983pt/> so that it discards irrelevant information. Ideally, <img src="https://rawgit.com/mostafarohani/info_bottle/master/svgs/2f118ee06d05f3c2d98361d9c30e38ce.svg?invert_in_darkmode" align=middle width=11.845020000000003pt height=22.381919999999983pt/> should extract the pertinent information to predict <img src="https://rawgit.com/mostafarohani/info_bottle/master/svgs/91aac9730317276af725abd8cef04ca9.svg?invert_in_darkmode" align=middle width=13.147200000000002pt height=22.381919999999983pt/>, but filter out extraneous details, which helps with generalization.

<img src="https://rawgit.com/mostafarohani/info_bottle/master/svgs/57c0d2c8c8834847a5b8414454adc686.svg?invert_in_darkmode" align=middle width=192.232095pt height=24.56552999999997pt/> and <img src="https://rawgit.com/mostafarohani/info_bottle/master/svgs/892b8c162360153e7575b208a7fd9889.svg?invert_in_darkmode" align=middle width=257.037495pt height=24.56552999999997pt/> (the network is deterministic, so <img src="https://rawgit.com/mostafarohani/info_bottle/master/svgs/0053c61aaf44c77507ae54dfaa2dae1d.svg?invert_in_darkmode" align=middle width=89.005455pt height=24.56552999999997pt/>).
They then discretize the input and activation spaces, and estimate the entropy using the empirical bin counts.
Using this procedure, their analysis is limited to toy networks and datasets (on order of 10s of inputs and features) so that the different possible values of <img src="https://rawgit.com/mostafarohani/info_bottle/master/svgs/43335f0bfad5de542db77de2c07893e4.svg?invert_in_darkmode" align=middle width=52.609425pt height=22.381919999999983pt/> can be enumerated during the entropy computation.

To investigate whether the phenomenon of the information bottleneck still appears for larger networks and real datasets, we employed *_deep generative models_* to estimate the density functions <img src="https://rawgit.com/mostafarohani/info_bottle/master/svgs/d9e560b0d39f8eb57862e63425a2a3d1.svg?invert_in_darkmode" align=middle width=32.830875pt height=24.56552999999997pt/> and <img src="https://rawgit.com/mostafarohani/info_bottle/master/svgs/2ed2140a9f85fc376bb493e3f7913ac8.svg?invert_in_darkmode" align=middle width=50.54065500000001pt height=24.56552999999997pt/>.
In particular, we used an autoregressive model styled after PixelCNN. Such methods factor the joint distribution <img src="https://rawgit.com/mostafarohani/info_bottle/master/svgs/48a18a027893eb4fb7f5352c2d3e89a4.svg?invert_in_darkmode" align=middle width=30.917205pt height=24.56552999999997pt/> over <img src="https://rawgit.com/mostafarohani/info_bottle/master/svgs/7e007cf84a2cddb9ef0c563dd8889f34.svg?invert_in_darkmode" align=middle width=52.210785pt height=22.473000000000006pt/> into a product of conditional distributions over its elements:
<img src="https://rawgit.com/mostafarohani/info_bottle/master/svgs/d6d9fed38385f70dc54bd46f60d582ef.svg?invert_in_darkmode" align=middle width=332.80054499999994pt height=26.401650000000007pt/>.
and then use a neural network to parametrize <img src="https://rawgit.com/mostafarohani/info_bottle/master/svgs/e9ce5471ff4e07901f4022ff5e3b9a99.svg?invert_in_darkmode" align=middle width=82.29243pt height=24.56552999999997pt/>. Given <img src="https://rawgit.com/mostafarohani/info_bottle/master/svgs/d774bd0d9f7d03409ea113c240e62c4a.svg?invert_in_darkmode" align=middle width=83.98681499999999pt height=14.102549999999994pt/>, the network outputs a distribution over <img src="https://rawgit.com/mostafarohani/info_bottle/master/svgs/9fc20fb1d3825674c6a279cb0d5ca636.svg?invert_in_darkmode" align=middle width=13.993485000000002pt height=14.102549999999994pt/> and is trained to maximize <img src="https://rawgit.com/mostafarohani/info_bottle/master/svgs/0d133d68c50f1a57d3077f15c81d1485.svg?invert_in_darkmode" align=middle width=54.889725pt height=24.56552999999997pt/>.

We trained a fully-connected network on the MNIST dataset, saved its activations at different points in the training process.
For each layer and iteration, we discretized each element of the activations into 32 bins, and trained two generative models. These modeled <img src="https://rawgit.com/mostafarohani/info_bottle/master/svgs/d9e560b0d39f8eb57862e63425a2a3d1.svg?invert_in_darkmode" align=middle width=32.830875pt height=24.56552999999997pt/> and <img src="https://rawgit.com/mostafarohani/info_bottle/master/svgs/2ed2140a9f85fc376bb493e3f7913ac8.svg?invert_in_darkmode" align=middle width=50.54065500000001pt height=24.56552999999997pt/>; the only difference being that the latter received the label <img src="https://rawgit.com/mostafarohani/info_bottle/master/svgs/91aac9730317276af725abd8cef04ca9.svg?invert_in_darkmode" align=middle width=13.147200000000002pt height=22.381919999999983pt/> as an additional input.
We then estimated <img src="https://rawgit.com/mostafarohani/info_bottle/master/svgs/04f24c77fd5784203a5948c61218523c.svg?invert_in_darkmode" align=middle width=152.42287499999998pt height=24.56552999999997pt/> and <img src="https://rawgit.com/mostafarohani/info_bottle/master/svgs/c998ddc6521d465ddbb4924c6e8791d9.svg?invert_in_darkmode" align=middle width=187.843095pt height=24.56552999999997pt/>, using the probabilities produced by our models and taking the expectation over the MNIST test set.

## Requirements
Specified in ```requirements.txt``` is the output of my ```pip freeze```, and therefore it has more packages than you will need to run this particular project. Regardless, you can get everything by running

```
pip install -r requirements.txt
```

You will still need to install CUDA and CUDNN to have tensorflow working on the GPU.

## Running the code

First you need to train an MNIST model and save the activations at different layers. Our Mnist model has 2 hidden layers, each with 256 hidden units and tanh nonlinearities. We consider the output layer, and after each activation function. We look at the values for iterations 0, 5, 10, 20, 1000, 9000. The code defaults to running on the 0th GPU. Run ```nvidia-smi``` to check which GPUs are avaliable on your machine.
```
python train_model.py --devices <GPU_NUM>
```
Now that you have the data, you can train the generator models. To speed up this process, we use MPI for parallelizing the workload. As an example, if GPUs 0,1,2,3 are avaliable, you would run
```
mpirun -np 4 python generator.py --devices 0,1,2,3
```

You can look at ```plots.ipynb``` for generating plots visualizing the results.

## Results

As a reference, this is the figure from Tishby and Zaslavky on their toy network.
<p align="center">
    <img src="assets/theirs.png" height="300">
</p>
And here is ours
<p align="center">
    <img src="assets/ours.png" height="300">
</p>

The two phases are apparent for <img src="https://rawgit.com/mostafarohani/info_bottle/master/svgs/ba4ad7d078f669d5c2c61a2eaf423293.svg?invert_in_darkmode" align=middle width=55.226655pt height=24.56552999999997pt/>, which increases and then decreases significantly for all layers. <img src="https://rawgit.com/mostafarohani/info_bottle/master/svgs/ab4b96f1e6312937210a8da7f5fa0b8d.svg?invert_in_darkmode" align=middle width=53.52963pt height=24.56552999999997pt/> generally increases, but there are a few outlying points from the overall trend. These could be attributed to inaccuracies in the fitted generative model, or could indicate that the information bottleneck phenomenon does not occur as prominently in deeper networks.
Although the results did not perfectly match the expected behavior, this the first (to our knowledge) analysis of the information bottleneck on nontrivial networks with nontrivial data.


