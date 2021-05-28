**[26]** Singing voice separation with deep U-Net - Jansson, A., Humphrey, E., Montecchio, N., Bittner, R., Kumar, A., & Weyde, T. (2017). Singing voice separation with deep U-Net convolutional networks. [Paper](https://openaccess.city.ac.uk/id/eprint/19289/1/)

Notes
=====

-	REImplementation from [Link](https://github.com/tsurumeso/vocal-remover/)

-	Analogous to image-to-image translation, where a mixed spectrogram is transformed into its constituent sources

-	Adapts the U-Net [24] architecture to the task of vocal separation.

-	In the reproduction of a natural image, displacements by just one pixel are usually not perceived as major distortions. In the frequency domain however, even a minor linear shift in the spectrogram has disastrous effects on perception: this is particularly relevant in music sig- nals, because of the logarithmic perception of frequency; moreover, a shift in the time dimension can become audi- ble as jitter and other artifacts. Therefore, it is crucial that the reproduction preserves a high level of detail.

-	The output of the final decoder layer is a soft mask that is multiplied element-wise with the mixed spectrogram to obtain the final estimate

-	The loss function used to train the model is the L1,1 norm 1 of the difference of the target spectrogram and the masked input spectrogram:

-	Downsample the input audio to 8192 Hz in order to speed up processing. We then com- pute the Short Time Fourier Transform with a window size of 1024 and hop length of 768 frames, and extract patches of 128 frames (roughly 11 seconds) that we feed as input and targets to the network. The magnitude spectrograms are normalized to the range [0, 1].

-	Two U-Nets, Θv and Θi, are trained to predict vocal and instrumental spectrogram masks, respectively.

-	A factor that we feel should be investigated further is the impact of large training data: work remains to be done to correlate the effects of the size of the training dataset to the quality of source separation.

-	Poor separation on tracks where the vocals are mixed at lower-than-average volume, uncompressed, suffer from extreme application of audio effects, or otherwise unconventionally mixed.

