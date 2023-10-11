import os
from pathlib import Path


# Use environment variables to auto-detect whether we are running an a Compute Canada cluster:
# Thanks to https://github.com/DM-Berger/unet-learn/blob/master/src/train/load.py for this trick.
COMPUTECANADA = False
TMP = os.environ.get("SLURM_TMPDIR")

if TMP:
    COMPUTECANADA = True

if COMPUTECANADA:
    INPUT_FOLDER = Path(str(TMP)).resolve() / "work" / "inputs"
    MASK_FOLDER = Path(str(TMP)).resolve() / "work" / "inputs" / "masks"
    PRETRAINED_MODEL_FOLDER = Path(str(TMP)).resolve() / "work" / "trained_models"
    PRETRAINED_MODEL_DDPM_PATH = (
        Path(str(TMP)).resolve() / "work" / "trained_models" / "ddpm"
    )
    PRETRAINED_MODEL_VAE_PATH = (
        Path(str(TMP)).resolve() / "work" / "trained_models" / "vae"
    )
    PRETRAINED_MODEL_DECODER_PATH = (
        Path(str(TMP)).resolve() / "work" / "trained_models" / "decoder"
    )
    PRETRAINED_MODEL_VGG_PATH = (
        Path(str(TMP)).resolve() / "work" / "trained_models" / "vgg16.pt"
    )
    OUTPUT_FOLDER = Path(str(TMP)).resolve() / "work" / "outputs"
else:
    INPUT_FOLDER = Path(__file__).resolve().parent.parent.parent / "data" / "IXI"
    MASK_FOLDER = Path(__file__).resolve().parent.parent / "masks"
    OASIS_FOLDER = Path(__file__).resolve().parent.parent.parent / "data" / "OASIS"
    PRETRAINED_MODEL_FOLDER = (
        Path(__file__).resolve().parent.parent.parent / "data" / "trained_models"
    )
    PRETRAINED_MODEL_DDPM_PATH = (
        Path(__file__).resolve().parent.parent.parent
        / "data"
        / "trained_models"
        / "ddpm"
    )
    PRETRAINED_MODEL_VAE_PATH = (
        Path(__file__).resolve().parent.parent.parent
        / "data"
        / "trained_models"
        / "vae"
    )
    PRETRAINED_MODEL_DECODER_PATH = (
        Path(__file__).resolve().parent.parent.parent
        / "data"
        / "trained_models"
        / "decoder"
    )
    PRETRAINED_MODEL_VGG_PATH = (
        Path(__file__).resolve().parent.parent.parent
        / "data"
        / "trained_models"
        / "vgg16.pt"
    )
    OUTPUT_FOLDER = (
        Path(__file__).resolve().parent.parent.parent / "data" / "outputs" / "ddpm"
    )
    THESIS_IMG_FOLDER = (
        Path(__file__).resolve().parent.parent.parent / "data" / "thesis_imgs"
    )
    FINAL_RESULTS_FOLDER = (
        Path(__file__).resolve().parent.parent.parent
        / "data"
        / "outputs"
        / "final_results"
    )

LATENT_SHAPE = [1, 3, 20, 28, 20]
IMAGE_SHAPE = [1, 1, 160, 224, 160]

IXI_IDs = [
    "002",
    "013",
    "015",
    "019",
    "022",
    "025",
    "027",
    "030",
    "031",
    "033",
    "034",
    "036",
    "039",
    "040",
    "041",
    "042",
    "043",
    "045",
    "046",
    "048",
    "049",
    "050",
    "051",
    "058",
    "059",
    "067",
    "068",
    "069",
    "072",
    "073",
    "074",
    "076",
    "077",
    "079",
    "080",
    "083",
    "087",
    "089",
    "090",
    "091",
    "092",
    "093",
    "095",
    "096",
    "097",
    "098",
    "099",
    "102",
    "104",
    "105",
    "108",
    "109",
    "110",
    "111",
    "114",
    "115",
    "118",
    "119",
    "120",
    "121",
    "122",
    "123",
    "126",
    "127",
    "128",
    "129",
    "130",
    "131",
    "132",
    "135",
    "140",
    "141",
    "146",
    "154",
    "156",
    "157",
    "158",
    "159",
    "160",
    "161",
    "162",
    "163",
    "164",
    "167",
    "168",
    "173",
    "174",
    "175",
    "176",
    "178",
    "179",
    "180",
    "181",
    "184",
    "186",
    "191",
    "192",
    "195",
    "197",
    "200",
]
