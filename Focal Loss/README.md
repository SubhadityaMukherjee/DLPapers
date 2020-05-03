**[19]** Focal Loss
Lin, T. Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017). Focal loss for dense object detection. In Proceedings of the IEEE international conference on computer vision (pp. 2980-2988).


## Notes
- Its good for huge imbalance apparently
-  one-stage detectors that are applied over a regular, dense sampling of possible object locations have the potential to be faster and simpler
- extreme foreground-background class imbalance encountered during training of dense detectors 
- We propose to address this class imbalance by reshaping the standard cross entropy loss such that it down-weights the loss assigned to well-classified examples.
-  focuses training on a sparse set of hard examples and prevents the vast number of easy negatives from overwhelm- ing the detector during training
- A common method for addressing class imbalance is to introduce a weighting factor α 
-  Easily classified negatives comprise the majority of the loss and dominate the gradient. While α balances the importance of positive/negative examples, it does not differentiate between easy/hard examples. Instead, we propose to reshape the loss function to down-weight easy examples and thus focus training on hard negatives.
- Binary classification models are by default initialized to have equal probability of outputting either y = −1 or 1. Under such an initialization, in the presence of class imbal- ance, the loss due to the frequent class can dominate total loss and cause instability in early training. To counter this, we introduce the concept of a ‘prior’ for the value of p es- timated by the model for the rare class (foreground) at the start of training. 
