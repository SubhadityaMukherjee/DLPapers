# Superresolution using an efficient sub-pixel convolutional neural network

- [Paper](https://arxiv.org/abs/1609.05158)
- Dataset used : [BSD300](http://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/BSDS300-images.tgz)
- Extract the dataset when you get it. Its around 30 mb


## How to run?

### Default
```bash
python3 main.py --epochs 10 --lr 0.01 --log-interval 100 --arch "my"
```
will run training

To run inference run
```bash
python3 example.py --input_image "./img.jpeg" --model "./models/model.pt" --output_filename "./outputs/output.jpg" --cuda 
```
## Video

To run for video. Run the inference code where input_image is the folder path of the video (in images)

Take the video and use ffmpeg

```bash
ffmpeg -i surprise.mp4 -r 13/1 $frame%04d.jpg   
```
13/1 is the frame rate

To make it back to video
```bash
ffmpeg -r 1/5 -i img%03d.jpg -c:v libx264 -vf fps=25 -pix_fmt yuv420p out.mp4

```

### Advanced run (IMPORTANT)

- Check the main.py file, you will find a list of all supported arguments there. Just follow the previous pattern and modify it to suit your purpose.
