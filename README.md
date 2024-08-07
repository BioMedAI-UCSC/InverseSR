![title](img/title.png)
---
We have developed an unsupervised technique for MRI super-resolution. We leverage a recent pre-trained Brain LDM for building powerful image priors over T1w brain MRIs. Our method is capable of being adapted to different settings of MRI SR problems at test time. Our method try to find the optimal latent representation $z^∗$ in the latent space of the Brain LDM, which could be mapped to represent the SR MRI $G(z^∗)$.

<p align="center">
    <img src="img/Method_Detail.gif" width="800" /> 
</p>

This gif shows the image space of the gradual optimization process when InverseSR finds the optimal latent representation $z^∗$.
<p align="center">
    <img src="img/InverseSR_2.gif" width="300" height="300" /> 
</p>

## Install Requirements
```sh
pip install -r requirements.txt
```

## Running InverseSR
We have given an example of ground truth high-resolution MRI `./inputs/ur_IXI022-Guys-0701-T1.nii.gz`. The code of generating low-resolution MRI is contained. Please download the Brain LDM parameters `ddpm` and `decoder` from [here](https://drive.google.com/drive/folders/110l68um6gUJzECIv0AyF-4Fcw0rrQgA9?usp=drive_link) into the InverseSR folder. Commands and parameters to run **InverseSR** can be found in `job_script/InverseSR(ddim).sh` and `job_script/InverseSR(decoder).sh` file.

## Data Preparation

### !! This model needs to be run on GPU/CPUs with at least 80GB of memory
You can find the necessary files for running the code [here](https://drive.google.com/drive/folders/110l68um6gUJzECIv0AyF-4Fcw0rrQgA9?usp=drive_link)