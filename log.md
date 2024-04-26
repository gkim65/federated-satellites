# Notes:

### Current Implementations

**April 25 2024**
Okay so I had done alot of things that made it difficult to run the github on my Mac but I have a temporary fix that lets me still use slurm in the process

Also **TODO** make sure to change the femnist data loading and google drive data loading/downloading into the correct files for personal computers if you are starting from scratch
**April 17-19 2024**
My god trying to understand and not mess up slrum test runs have been a nightmare...
But I think I finally got to a place where tests are running relatively smoothly?

Todos however:
- need to check how to fix the idle time and the idle average
- is my thing actually fed avg? or a type of fed prox? please check
    - Okay I checked, I just did a slightly more efficient method of fedAvg, might need to make fedAvg worse and change it for the future

**April 12-16 2024**

Many things have happened over the past few days

*FL SHENANIGANS:*
- I think I have fed_avg fully set up omg im so happy
- but several important changes:
    - now the simulation can keep running new sims even if one sim crashes, which happens if i get to the end of the csv but don't have enough entries
    - I also only take in info from one csv for all things I test within the groundstations and different satellite clusters and other things
    - For fed avg, we now wait till it gets all of the satellites to finish in a round so each duration in a round takes much longer, I should try and track this as well for future FL implementations >> fed prox we only have specific numbers of hours we wait each round doesn't matter if we dont have all clients 
    - for fed avg, don't worry about which groundstation or satellties cause we wait for all of them (maybe i should save what groundstation each pass is from but eh)
        - **FOR FEDPROX SAVE WHICH GROUNDSTATION**
    - Lots of new outputs saved for thingss to plot:
        - accuracy (as always) but change to get final accuracy later and plot this across all runs!
        - Idle time **This coudl get really interesting and also help justify why we need one alg over another**
        - Duration **Also duration**
        - Can also track how long each pass is backwards using the duration and see if we can actually run as many epochs as we want to in the time that is available to us
        - Coverage map **TODO maybe with STK?**
        


*STK SHENANIGANS:*
- I realize that the way I was saving my stk files were alot more cumbersome and much more difficult than it needed to be, so there is a small solution :D *not sure if this is the best way to run simms in the future, but this seemed to make the most sense while I was trying out different methods*
    - So instead of running different sims for every single parameter (there is no way to automate this to my knowledge, I just have to keep clicking which took me a whole day and im not doing that again) I just make one huge sim that takes a few hours to do but if it runs it creates a huge CSV file that I keep on google drive
    - I can pick and choose which satellites and groundstations I want to test out using this huge file, so this huge simulation file can last me for multitudes of experiments and I only have to run this a few times every time I do a batch experiment

Notes for grace in the future as she figures out how to run these simulations (if you forget again)

1) Make your simulation by first setting the groundstations you want to test (The main sim should have all of this set up already for groundstations in the US and Landsat, but if you ever want to test different place locations its possible to keep adding >> the landsat groundstation layout may be at the limit of what STK will be willing to run on my computer though) **After making a set of groundstations to work with, put them in a constellation object**
2) make a satellite, (I've just done one with a polar orbit so far with 90 deg inclination and 500km alt) and make more bb satellties from that satellite by doing a walker star formation, i think the max I'm willing to wait for my computer to not crash is 50 satellties and 25 clusters but you can try to push it further **Add this also to a constellation object**
3) make a chain, and add in SATELLITE constellation first, then add GROUNDSTATION constellation (i want satellties to make first contact for all instances)
4) right click chain, then reports and access, then get access data :D



**April 1st 2024**
- revisited github page
- Things implemented so far:
    - Trying to ensure that the different datasets can be easily switched between; set up a `FEMNIST_tests` folder that has all the models and functionality for FEMNISt datasets
    - Weights and biases are now implemented for basic info (config parameters, along with accumulated losses and accuracies) didn't realize `wandb` stood for this lol
    - `config_maker.py` provides an easy way to make config files to just run over for different parameter sweeps
- trying to run on cluster
    - realizing that ray can only be downloaded on python 3.10 so need to set taht for environments in the future
- TODOS:
    - implement `CIFAR-10` and `CIFAR-100` datasets in separate folders
    - `client_fn` can't take in any inputs so I can't force it to take in config parameter to run a specific dataset... right now, still manual but can probably find a way around it later
    


**March 27th 2024**
- Set up on the GPU Cluster, both HPC and CamMLSys
- wrote up how to set up conda for CamMLSys cluster
