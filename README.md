# human-skills-encoding-and-motion-reproduction
This package gives an example of how the human skills can be captured via human demonstraions, encoded by GMM-HMM algorithm and then
Reproduced in new situations.  

This work followed the Robot Learning from Demonstration framwwork and built upon exiting HMM library.  The data are come from a Peg-in-hole (PiH) example where a human opertor's hand motion is recorded using Vicon system and the corresponding force signal is recorded using Force Torque sensor.

The processing flow is briefly introduced below:

1) Mark the start and end point of the PiH in the FT signal profile and save it into curser.mat file.

2) Preprocessing (e.g. resampling and denoise) and structure the input data.

3) Gaussian Mixture Model and Hidden Markov Model encoding; return the model parameters for P(X;theta) where X is the observations.

4) Using the model and reproduce the motion using a mass-damper imepedence controller.

5) Plot the results.
