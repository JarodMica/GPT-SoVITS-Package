# Jarod's NOTE
Working on turning this into a package.  Right now, the API *does in fact* work to make requests to and this can be installed.

## Quick Install and Usage
Ideally, do this all inside of a venv for package isolation
1. Install by doing:
  ```
  pip install git+https://github.com/JarodMica/GPT-SoVITS-Package.git
  ```
2. Make sure torch is installed with CUDA enabled.  Reccomend to run `pip uninstall torch` to uninstall torch, then reinstall with the following.  I chose 2.4.0+cu121:
  ```
  pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121
  ```

Now to use it, so far I've only tested it with the api_v2.py.  Given that the install above went fine, you should now be able to run:
```
gpt_sovits_api
```
Which will bootup local server that you can make requests to.  Checkout `test.py` and `test_streaming.py` to get an idea for how you might be able to use the API.

## Pretrained Models
Probably don't need to follow the instructions for the below, these are just kept here for reference for now.

1. Download pretrained models from [GPT-SoVITS Models](https://huggingface.co/lj1995/GPT-SoVITS) and place them in `GPT_SoVITS/pretrained_models`.

2. Download G2PW models from [G2PWModel_1.1.zip](https://paddlespeech.bj.bcebos.com/Parakeet/released_models/g2p/G2PWModel_1.1.zip), unzip and rename to `G2PWModel`, and then place them in `GPT_SoVITS/text`.(Chinese TTS Only)

3. For UVR5 (Vocals/Accompaniment Separation & Reverberation Removal, additionally), download models from [UVR5 Weights](https://huggingface.co/lj1995/VoiceConversionWebUI/tree/main/uvr5_weights) and place them in `tools/uvr5/uvr5_weights`.

    - If you want to use `bs_roformer` or `mel_band_roformer` models for UVR5, you can manually download the model and corresponding configuration file, and put them in `tools/uvr5/uvr5_weights`. **Rename the model file and configuration file, ensure that the model and configuration files have the same and corresponding names except for the suffix**. In addition, the model and configuration file names **must include `roformer`** in order to be recognized as models of the roformer class.

    - The suggestion is to **directly specify the model type** in the model name and configuration file name, such as `mel_mand_roformer`, `bs_roformer`. If not specified, the features will be compared from the configuration file to determine which type of model it is. For example, the model `bs_roformer_ep_368_sdr_12.9628.ckpt` and its corresponding configuration file `bs_roformer_ep_368_sdr_12.9628.yaml` are a pair, `kim_mel_band_roformer.ckpt` and `kim_mel_band_roformer.yaml` are also a pair.

4. For Chinese ASR (additionally), download models from [Damo ASR Model](https://modelscope.cn/models/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/files), [Damo VAD Model](https://modelscope.cn/models/damo/speech_fsmn_vad_zh-cn-16k-common-pytorch/files), and [Damo Punc Model](https://modelscope.cn/models/damo/punc_ct-transformer_zh-cn-common-vocab272727-pytorch/files) and place them in `tools/asr/models`.

5. For English or Japanese ASR (additionally), download models from [Faster Whisper Large V3](https://huggingface.co/Systran/faster-whisper-large-v3) and place them in `tools/asr/models`. Also, [other models](https://huggingface.co/Systran) may have the similar effect with smaller disk footprint.

## Dataset Format

The TTS annotation .list file format:

```
vocal_path|speaker_name|language|text
```

Language dictionary:

- 'zh': Chinese
- 'ja': Japanese
- 'en': English
- 'ko': Korean
- 'yue': Cantonese

Example:

```
D:\GPT-SoVITS\xxx/xxx.wav|xxx|en|I like playing Genshin.
```

## Finetune and inference

### Open WebUI

#### Integrated Package Users

Double-click `go-webui.bat`or use `go-webui.ps1`
if you want to switch to V1,then double-click`go-webui-v1.bat` or use `go-webui-v1.ps1`

#### Others

```bash
python webui.py <language(optional)>
```

if you want to switch to V1,then

```bash
python webui.py v1 <language(optional)>
```
Or maunally switch version in WebUI

### Finetune

#### Path Auto-filling is now supported

    1. Fill in the audio path
    2. Slice the audio into small chunks
    3. Denoise(optinal)
    4. ASR
    5. Proofreading ASR transcriptions
    6. Go to the next Tab, then finetune the model

### Open Inference WebUI

#### Integrated Package Users

Double-click `go-webui-v2.bat` or use `go-webui-v2.ps1` ,then open the inference webui at  `1-GPT-SoVITS-TTS/1C-inference`

#### Others

```bash
python GPT_SoVITS/inference_webui.py <language(optional)>
```
OR

```bash
python webui.py
```
then open the inference webui at `1-GPT-SoVITS-TTS/1C-inference`

## V2 Release Notes

New Features:

1. Support Korean and Cantonese

2. An optimized text frontend

3. Pre-trained model extended from 2k hours to 5k hours

4. Improved synthesis quality for low-quality reference audio

    [more details](https://github.com/RVC-Boss/GPT-SoVITS/wiki/GPT%E2%80%90SoVITS%E2%80%90v2%E2%80%90features-(%E6%96%B0%E7%89%B9%E6%80%A7))

Use v2 from v1 environment:

1. `pip install -r requirements.txt` to update some packages

2. Clone the latest codes from github.

3. Download v2 pretrained models from [huggingface](https://huggingface.co/lj1995/GPT-SoVITS/tree/main/gsv-v2final-pretrained) and put them into `GPT_SoVITS\pretrained_models\gsv-v2final-pretrained`.

    Chinese v2 additional: [G2PWModel_1.1.zip](https://paddlespeech.bj.bcebos.com/Parakeet/released_models/g2p/G2PWModel_1.1.zip)（Download G2PW models,  unzip and rename to `G2PWModel`, and then place them in `GPT_SoVITS/text`.

## V3 Release Notes

New Features:

1. The timbre similarity is higher, requiring less training data to approximate the target speaker (the timbre similarity is significantly improved using the base model directly without fine-tuning).

2. GPT model is more stable, with fewer repetitions and omissions, and it is easier to generate speech with richer emotional expression.

    [more details](https://github.com/RVC-Boss/GPT-SoVITS/wiki/GPT%E2%80%90SoVITS%E2%80%90v3%E2%80%90features-(%E6%96%B0%E7%89%B9%E6%80%A7))

Use v3 from v2 environment:

1. `pip install -r requirements.txt` to update some packages

2. Clone the latest codes from github.

3. Download v3 pretrained models (s1v3.ckpt, s2Gv3.pth and models--nvidia--bigvgan_v2_24khz_100band_256x folder) from [huggingface](https://huggingface.co/lj1995/GPT-SoVITS/tree/main) and put them into `GPT_SoVITS\pretrained_models`.

    additional: for Audio Super Resolution model, you can read [how to download](./tools/AP_BWE_main/24kto48k/readme.txt)


## Todo List

- [x] **High Priority:**

  - [x] Localization in Japanese and English.
  - [x] User guide.
  - [x] Japanese and English dataset fine tune training.

- [ ] **Features:**
  - [x] Zero-shot voice conversion (5s) / few-shot voice conversion (1min).
  - [x] TTS speaking speed control.
  - [ ] ~~Enhanced TTS emotion control.~~ Maybe use pretrained finetuned preset GPT models for better emotion.
  - [ ] Experiment with changing SoVITS token inputs to probability distribution of GPT vocabs (transformer latent).
  - [x] Improve English and Japanese text frontend.
  - [ ] Develop tiny and larger-sized TTS models.
  - [x] Colab scripts.
  - [x] Try expand training dataset (2k hours -> 10k hours).
  - [x] better sovits base model (enhanced audio quality)
  - [ ] model mix

## (Additional) Method for running from the command line
Use the command line to open the WebUI for UVR5
```
python tools/uvr5/webui.py "<infer_device>" <is_half> <webui_port_uvr5>
```
<!-- If you can't open a browser, follow the format below for UVR processing,This is using mdxnet for audio processing
```
python mdxnet.py --model --input_root --output_vocal --output_ins --agg_level --format --device --is_half_precision
``` -->
This is how the audio segmentation of the dataset is done using the command line
```
python audio_slicer.py \
    --input_path "<path_to_original_audio_file_or_directory>" \
    --output_root "<directory_where_subdivided_audio_clips_will_be_saved>" \
    --threshold <volume_threshold> \
    --min_length <minimum_duration_of_each_subclip> \
    --min_interval <shortest_time_gap_between_adjacent_subclips>
    --hop_size <step_size_for_computing_volume_curve>
```
This is how dataset ASR processing is done using the command line(Only Chinese)
```
python tools/asr/funasr_asr.py -i <input> -o <output>
```
ASR processing is performed through Faster_Whisper(ASR marking except Chinese)

(No progress bars, GPU performance may cause time delays)
```
python ./tools/asr/fasterwhisper_asr.py -i <input> -o <output> -l <language> -p <precision>
```
A custom list save path is enabled

## Credits

Special thanks to the following projects and contributors:

### Theoretical Research
- [ar-vits](https://github.com/innnky/ar-vits)
- [SoundStorm](https://github.com/yangdongchao/SoundStorm/tree/master/soundstorm/s1/AR)
- [vits](https://github.com/jaywalnut310/vits)
- [TransferTTS](https://github.com/hcy71o/TransferTTS/blob/master/models.py#L556)
- [contentvec](https://github.com/auspicious3000/contentvec/)
- [hifi-gan](https://github.com/jik876/hifi-gan)
- [fish-speech](https://github.com/fishaudio/fish-speech/blob/main/tools/llama/generate.py#L41)
- [f5-TTS](https://github.com/SWivid/F5-TTS/blob/main/src/f5_tts/model/backbones/dit.py)
- [shortcut flow matching](https://github.com/kvfrans/shortcut-models/blob/main/targets_shortcut.py)
### Pretrained Models
- [Chinese Speech Pretrain](https://github.com/TencentGameMate/chinese_speech_pretrain)
- [Chinese-Roberta-WWM-Ext-Large](https://huggingface.co/hfl/chinese-roberta-wwm-ext-large)
- [BigVGAN](https://github.com/NVIDIA/BigVGAN)
### Text Frontend for Inference
- [paddlespeech zh_normalization](https://github.com/PaddlePaddle/PaddleSpeech/tree/develop/paddlespeech/t2s/frontend/zh_normalization)
- [split-lang](https://github.com/DoodleBears/split-lang)
- [g2pW](https://github.com/GitYCC/g2pW)
- [pypinyin-g2pW](https://github.com/mozillazg/pypinyin-g2pW)
- [paddlespeech g2pw](https://github.com/PaddlePaddle/PaddleSpeech/tree/develop/paddlespeech/t2s/frontend/g2pw)
### WebUI Tools
- [ultimatevocalremovergui](https://github.com/Anjok07/ultimatevocalremovergui)
- [audio-slicer](https://github.com/openvpi/audio-slicer)
- [SubFix](https://github.com/cronrpc/SubFix)
- [FFmpeg](https://github.com/FFmpeg/FFmpeg)
- [gradio](https://github.com/gradio-app/gradio)
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
- [FunASR](https://github.com/alibaba-damo-academy/FunASR)
- [AP-BWE](https://github.com/yxlu-0102/AP-BWE)

Thankful to @Naozumi520 for providing the Cantonese training set and for the guidance on Cantonese-related knowledge.

## Thanks to all contributors for their efforts

<a href="https://github.com/RVC-Boss/GPT-SoVITS/graphs/contributors" target="_blank">
  <img src="https://contrib.rocks/image?repo=RVC-Boss/GPT-SoVITS" />
</a>
