![title](img/title.png)
---
Implementation of paper "InverseSR: 3D Brain MRI Super-Resolution Using a Latent Diffusion Model" of Jueqi Wang, Jacob Levman, Walter Hugo Lopes Pinaya, Petru-Daniel Tudosiu, M. Jorge Cardoso and Razvan Marinescu, in MICCAI 2023.


## Install Requirements
```sh
pip install -r requirements.txt
```

## Running InverseSR
We have given an example of ground truth high-resolution MRI image `./inputs/IXI_T1_069.pth`. Commands and parameters to run **InverseSR** can be found in `job_script/InverseSR(ddim).sh` and `job_script/InverseSR(decoder).sh` file.


## Data Preparation

### !! This model needs to be run on GPU/CPUs with at least 80GB of memory
You can find the necessary files for running the code [here](https://drive.google.com/drive/folders/110l68um6gUJzECIv0AyF-4Fcw0rrQgA9?usp=drive_link)