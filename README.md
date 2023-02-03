# Applied Data Science Project
<div id="top"></div>

<br />
<div align="center">
    <p float="left">
    <img src="logo/logo_post.png" width="300", height="300" />
    <img src="logo/logo_mask.png" width="300", height="300" /> 
    </p>

  <h3 align="center">Weakly supervised burnt area discrimination pipeline <br /> PoliTo 2022/2023</h3>

  <p align="center">
    In the following, the whole project is briefly presented in how it works and how the various scripts should be executed to reproduce our results.
  </p>
</div>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#about-the-project">About The Project</a></li>
    <li><a href="#preprocessing">Preprocessing engine</a></li>
    <li><a href="#tests">Tests</a></li>
    <li><a href="#presentation">Presentation Instruction</a></li>
    <li><a href="#interface">Train your own model: interface.py</a></li>
  </ol>
</details>

<!-- ABOUT THE PROJECT -->
## About The Project
This project is about Image Segmentation, a machine learning task which consists in delineating specific shapes and patterns in input images. To produce trained models for further use, please referd to run_nn.sh or test_*.py; make sure to create the folders models/ and models/trained_models/ first. To see an example of segmentation, please refer to presentation_notebook.ipynb, in which all the necessary steps (if a suitably trained model is provided) are contained.

<p align="right">(<a href="#top">back to top</a>)</p>

## Preprocessing
The preprocessing engine handles the input datasets selected. datasetscanner.py and datasetformatter.py take care of producing logs necessary for effisdataset.py and colombadataset.py in order for the dataloaders to work properly.

<p align="right">(<a href="#top">back to top</a>)</p>

## Tests
test_*.py files handle the training or fine-tuning of a U-Net over specific datasets/foundation models, if one wishes to produce results all at once, run_nn.sh will do the job. At the current state of the implementation no pre-trained models are present and will be added with suitable sources in later updates.

<p align="right">(<a href="#top">back to top</a>)</p>

## Presentation
The Jupyter Notebook presentation_notebook.ipynb contains just a live demostration of the complete pipeline, from tiling, through predictions and eventually assembling of the final activation. (Smooth merging to be added on later releases.)

<p align="right">(<a href="#top">back to top</a>)</p>

## Interface
If one wishes to train, validate and test a specific model over a selected dataset (whose paths have to be appropriately configured  in config/config.json), interface.py comes in handy. Please run
```sh
python3 interface.py --help
```
for a set of instruction regarding the possible parameters to be used.

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- CONTACTS -->
## CONTACTS
Francesco Pagano  s299266@studenti.polito.it <br />
Sofia Fanigliulo  s300751@studenti.polito.it <br />
Giuseppe Esposito  s302179@studenti.polito.it <br />
Aurora Gensale  s303535@studenti.polito.it <br />
Elena Di Felice s303499@studenti.polito.it <br />