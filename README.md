# FinalProject_iannwtf20_21_Group11
Final project of Group 11 for the Implementing Artificial Neural Networks with Tensorflow course 2020/2021

## How to create your own dataset
with the provided scripts you can fetch videos from youtube based on category and build datasets from them that contains a sets of images together with the respective audio clips.<br>

**1. step:** run fetch_yt_videos.py script<br>
adjustable parameters: *categories, filters*

> this will create the folder *videos/* and subfolders for each *category*
> in the subfolders up to ~30 (often less) videos from youtube will be downloaded.<br>

> before fetching the videos will be filtered if they have to many views are too long or have the term "music" in their description.

**2. step:** run videos_to_dataset.py script<br>
adjustable parameters: *clip_length, fps, iamge sizes, sample_rate*

> all videos are first split into clips of length *clip_length* then frames are extracted according to *fps*.<br>

> the frames are stored in *dataset_images.pbz2*<br>
> with shape *(clip_num, frame_num, image_height, image_width, channels)*

> the audio is stored in *dataset_audio.pbz2*<br>
> with shape *(clip_num, 1, clip_length * sample_rate)*

## How to load your custom dataset
```python
def decompress_pickle(file):
    data = bz2.Bz2File(file, "rb")
    data = pickle.load(data)
    return data
```