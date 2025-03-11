<div align="center">

# NEUTART: NEUral Text to ARticulate Talk

Official implementation of "Neural Text to Articulate Talk: Deep Text to Audiovisual Speech Synthesis achieving both Auditory and Photo-realism". 

[![Paper](https://img.shields.io/badge/arXiv-2312.06613-brightgreen)](https://arxiv.org/abs/2312.06613)
&nbsp; [![Project WebPage](https://img.shields.io/badge/Project-webpage-blue)](https://g-milis.github.io/neutart.html)
&nbsp; <a href="https://youtu.be/W2DNiI0j5a8"><img src="https://img.shields.io/badge/Youtube-Video-red?style=flat&logo=youtube&logoColor=red" alt="Youtube Video"></a>

</div>


## Installation
System requirements: Linux, a GPU, `conda`, and up to 10 GB of disk space (you may adjust the paths to write to an external hard drive).

Clone the repository:
```bash
git clone https://github.com/g-milis/NEUTART && cd NEUTART
```
Create a virtual environment and use it for all the commands in this guide, unless specified otherwise:
```bash
conda env create -f environment.yml && conda activate NEUTART
```
Before downloading the pretrained models, you need to create a [FLAME](https://flame.is.tue.mpg.de/) account (the setup script will ask for your credentials). Then, download the required assets and pretrained models with:
```bash
./setup.sh
```
The script downloads only the missing files, so it can rerun if necessary.

Due to licensing reasons, you have to follow the next two installation steps manually. Do not worry, they provide detailed instructions.

The photorealistric preprocessing step employs the [FSGAN](https://github.com/YuvalNirkin/fsgan) face segmenter, which requires a simple [form](https://docs.google.com/forms/d/e/1FAIpQLScyyNWoFvyaxxfyaPLnCIAxXgdxLEMwR9Sayjh3JpWseuYlOA/viewform) to obtain it. Please follow the instructions to download `lfw_figaro_unet_256_2_0_segmentation_v1.pth` (from the `v1` folder) and place it under `photorealistic/preprocessing/segmentation`.

Finally, you need to create a texture model with [BFM_to_FLAME](https://github.com/TimoBolkart/BFM_to_FLAME#create-texture-model). Please follow the instructions to create the model `FLAME_albedo_from_BFM.npz` and place it under `spectre/data`.


## Inference
In the `assets` directory you will find reference videos for two TCD-TIMIT subjects. Note that the default one for inference is `21M`. Process the reference video with:
```bash
./photorealistic/preprocess.sh test processed_videos/21M_test assets/21M
```
Now run:
```bash
./inference.sh
```
You may easily adjust the inference parameters inside the script. Also see `misc/batch_inference.sh`. If you want to run inference after training on your own subjects, make sure to reserve a few-second clip to use as reference.


## Training
### Photorealistic
We will finetune the pre-trained renderer in `checkpoints/meta-renderer`. You need 5-10 minutes of identically lit training clips of your subject. You can find out their total duration with `./misc/duration.sh`. Suppose you have the clips in `assets/09F_train`. Process them with:
```bash
./photorealistic/preprocess.sh train <SUBJECT_DIR> assets/09F_train
```
where `<SUBJECT_DIR>` is the directory where the processing outputs will be saved, for instance `processed_videos/09F_train`. Please note that the photorealistic preprocessing has to precede the audiovisual preprocessing.

Train the renderer with `./photorealistic/train.sh <SUBJECT_NAME> <SUBJECT_DIR>`, as in:
```bash
./photorealistic/train.sh 09F processed_videos/09F_train
```
You may adjust the training script according to the options in `photorealistic/renderer/options`.


### Audiovisual
You need to download the [Montreal Forced Aligner](https://montreal-forced-aligner.readthedocs.io/en/latest/) (MFA):
```bash
conda install montreal-forced-aligner
```
If it's causing any trouble, just create a dedicated environment for `mfa` commands:
```bash
conda create -n mfa && conda activate mfa && conda config --add channels conda-forge && conda install montreal-forced-aligner
```

Once installed, run with the `mfa` environment:
```bash
mfa model download acoustic english_us_arpa
```

Let's finetune the pre-trained multispeaker model in `checkpoints/TCD-TIMIT` on clips of a single speaker. Suppose you have a set of `.mp4` clips along with their transcriptions. We will use TCD-TIMIT subject `09F` as an example.

1. Copy the contents of `audiovisual/config/21M` to `audiovisual/config/09F` and adjust `preprocess.yaml` according to the prompts. The scripts with a `-d` argument expect the directory name of your dataset's configs, under `config` (e.g. `21M` and `09F`).

2. The MFA expects a file structure `<DATASET>/<SPEAKER>/<UTTERANCE>.(wav|lab)`, in our case `09F/09F/*.(wav|lab)` (in the single speaker case, the dataset and the speaker coincide). You can either construct it manually, or run `audiovisual/prepare_align.py` after modifying it accordingly:
```bash
python audiovisual/prepare_align.py -d 09F
```

3. Perform the text-to-audio alignment. Use the `<TTS_PATH>` and `<PROCESSED_PATH>` that you specified in `preprocess.yaml` (e.g. `tts_data/09F` and `processed_data/09F`):
```bash
mfa align <TTS_PATH> audiovisual/text/librispeech-lexicon.txt english_us_arpa <PROCESSED_PATH>
```

4. Preprocess the aligned data using the script below. Bear in mind that the preprocessor expects the videos in the format `<UTTERANCE>.mp4`. If you don't have this format, you can just adjust the `video_path` variable in `Preprocessor.build_from_path()`:
```bash
python audiovisual/preprocess.py -d 09F
```

5. Adjust `audiovisual/config/09F/train.yaml` according to the prompts and train the audiovisual module with:
```bash
python audiovisual/train.py -d 09F
```

## In-the-wild
For training on in-the-wild videos, make sure you select a subject with adequate training footage, neutral head pose, and consistnet lighting throughout the video. We will demonstrate on a [sample](https://www.youtube.com/watch?v=jSw_GUq6ato) of the [HDTF](https://github.com/MRzzm/HDTF/blob/main/HDTF_dataset/WRA_video_url.txt#L4) dataset.

1. Download the video with:
```bash
yt-dlp -f "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best" -o assets/HDTF_raw.mp4 https://www.youtube.com/watch?v=jSw_GUq6ato
```

2. Change fps to 25 (recommended) and convert to a `wav` file:
```bash
ffmpeg -i assets/HDTF_raw.mp4 -r 25 -c:v libx264 assets/HDTF.mp4
```
```bash
ffmpeg -i assets/HDTF.mp4 assets/HDTF.wav
```

3. Transcribe it using the system of your choice. You may find this [Whisper API](https://replicate.com/openai/whisper) useful, since it also splits the speech into smaller utterances. Save the resulting JSON file and provide it as an input to the next step.

4. Use the file `misc/split_video.py` and change the parameters at the top to match your configuration. Then, run the python file to split your video into clips. From then on, you can move forward by following the generic steps above.


## Troubleshooting
- Please inspect the output of `./photorealistic/preprocess.sh`, making sure the images look consistent.
- If your subject is not from TCD-TIMIT, you might have to adjust the parameters in `audiovisual.utils.lipread.cut_mouth` before preprocessing, in order to write the mouth clips. The videos need to be cropped from the bottom of the nose to the chin, with the mouth centered, like `misc/mouth.mp4` (you may comment out the line `Normalize(mean, std)` for easier inspection, but please reinclude it).
- The default training configuration updates all the weights in the audiovisual module, but specifying `transfer=True` allows you to freeze the encoder weights if you want to train on few data.
- Examine the training progress of the audiovisual module with `tensorboard --logdir checkpoints/<DATASET>/audiovisual/log`.
- If the output of the audiovisual module is jittery, please increase the `weight_vector` for the regularization loss `exp_reg_loss` in `audiovisual.model.loss.AudiovisualLoss.forward`.
- Examine the training progress of the photorealistic module by opening `checkpoints/09F/photorealistic/web/index.html`.


## Note on Social Impact
Deep learning systems for photorealistic talking head generation like NEUTART can have a very positive impact in many applications such as digital avatars, virtual assistants, accessibility tools, teleconferencing, video games, movie dubbing, and human-machine interfaces. However, this type of technology has the risk of being misused towards unethical or malicious purposes, since it can produce deepfake videos of individuals without their consent. We believe that researchers and engineers working in this field need to be mindful of these ethical issues and contribute to raising public awareness about the capabilities of such AI systems, as well as the development of state-of-the-art deepfake detectors. In our work, generated videos are always presented as synthetic, either explicitly or implicitly (when clearly implied by the context), and we encourage you to follow this practice. Please make sure that you understand the conditions under which this project is licensed, before using it.


## Acknowledgements
Our code is based on these great projects:
- [FastSpeech2](https://github.com/ming024/FastSpeech2)
- [NED](https://github.com/foivospar/NED)
- [spectre](https://github.com/filby89/spectre)
- [Visual_Speech_Recognition_for_Multiple_Languages](https://github.com/mpc001/Visual_Speech_Recognition_for_Multiple_Languages)


## Contact
Please do not hesitate to raise issues, create pull requests, or ask questions via [email](mailto:milis27400@gmail.com). I will reply swiftly.


## Citation
If you find this work useful for your research, please cite our paper:
```
@misc{milis2023neural,
  title={Neural Text to Articulate Talk: Deep Text to Audiovisual Speech Synthesis achieving both Auditory and Photo-realism},
  author={Milis, Georgios and Filntisis, Panagiotis P. and Roussos, Anastasios and Maragos, Petros},
  journal={arXiv preprint arXiv:2312.06613},
  year={2023}
}
```
