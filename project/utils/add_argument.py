from argparse import ArgumentParser
from html import parser

import torch

def add_argument(parser: ArgumentParser):
    parser.add_argument(
        "--crop_top_y", 
        type=int, 
        default=0, 
        help="Angolo Y superiore da cui ritagliare il cubo grande"
        )
    parser.add_argument(
        "--crop_left_x", 
        type=int, 
        default=0, 
        help="Angolo X sinistro da cui ritagliare il cubo grande"
        )
    parser.add_argument(
        "--crop_h", 
        type=int, 
        default=128, 
        help="Altezza del crop (dimensione di training)"
        )
    parser.add_argument(
        "--crop_w", 
        type=int, 
        default=128, 
        help="Larghezza del crop (dimensione di training)"
        )
    # --- LOGGING & PATHS ---
    parser.add_argument(
                "--patience",
                type=int,
                default=20,
                help="Patience for Early stop optimization",
            )
    parser.add_argument(
                    "--min_delta",
                    type=float,                    
                    default=1e-4,
                    help = "Threshold for early stop optimization",
                )
    parser.add_argument(
            "--hr_path",
            # Modificato: Path generico per astro
            default=r"./data/inputs/Data_mix_OK/test/patch_000033.npy",
            help="Dir per i dati di input dell'inversione",
        )
    parser.add_argument(
                "--lr_path",
                # Modificato: Path generico per astro
                default=r"./data/inputs/16x128x128_cont_ldev_OK/test_LR/patch_000000.npy",
                help="Path per patch LR",
            )
    parser.add_argument(
        "--tensor_board_logger_decoder",
        # Modificato: Path generico per astro
        default=r"./logs/BRGM_decoder",
        help="Dir per i log di TensorBoard",
    )

    parser.add_argument(
        "--tensor_board_logger_ddim",
        # Modificato: Path generico per astro
        default=r"./logs/BRGM_ddim",
        help="Dir per i log di TensorBoard",
    )

    parser.add_argument(
        "--norm_data",
        # Modificato: Path generico per astro
        default="local",
        help="Modalità di normalizzazione dei dati",
    )
    
    # --- DATI E FORMATI ---
    parser.add_argument(
        "--data_format",
        default="npy", 
        type=str,
        choices=["fits", "txt", "npy"],
        help="Formato del datacube di input"
    )
    parser.add_argument(
        "--inference",
        action="store_true"
    )
    parser.add_argument(
        "--test_mode",
        action="store_true"
    )
    parser.add_argument(
        "--object_id", # Modificato: da subject_id a object_id
        default="patch_000000.npy",
        type=str,
        help="ID dell'oggetto celeste (es. nome galassia o ID catalogo)"
    )
    
    parser.add_argument(
    "--image_size", 
        type=int, 
        nargs='+', 
        default=[128, 128, 128], 
        help="Risoluzione target (D, H, W)"
    )

    parser.add_argument(
    "--z_channels",
        type=int,
        default=3,
        help="Canali del latente"
    )

    parser.add_argument(
    "--out_channels",
        type=int,
        default=1,
        help="Canali in input nel vgg e in output dal decoder"
    )
    # --- PARAMETRI DI TRAINING GENERICI ---
    parser.add_argument(
        "--learning_rate",
        dest="learning_rate",
        default=1e-4, # Spesso astro richiede LR più bassi
        type=float,
    )
    parser.add_argument(
        "--batch_size",
        default=1, # I datacube 3D occupano molta VRAM
        type=int,
    )
    parser.add_argument(
        "--seed",
        default=42,
        type=int,
    )
    
    # --- OTTIMIZZAZIONE LATENTE (INVERSION) ---
    parser.add_argument(
        "--experiment_name",
        default="astro_restoration",
        type=str,
    )
    parser.add_argument(
        "--update_latent_variables",
        action="store_true",
        help="Se attivo, ottimizza il vettore latente z"
    )
    parser.add_argument(
        "--update_conditioning",
        action="store_true",
        help="Se attivo, ottimizza le condizioni fisiche (es. redshift)"
    )
    
    # --- CONDIZIONAMENTO ASTROFISICO ---
    # 'hi_size', 'line_flux_integral', 'i', 'w20'
    parser.add_argument(
        "--update_hi_size",
        action="store_true",
    )
    parser.add_argument(
        "--update_line_flux_integral",
        action="store_true",
    )
    parser.add_argument(
        "--update_i",
        action="store_true",
    )
    parser.add_argument(
        "--update_w20",
        action="store_true",
    )
  

    # --- PERCEPTUAL LOSS & GEOMETRIA ---
    parser.add_argument(
        "--lambda_perc",
        default=0.01,
        type=float,
    )

    parser.add_argument(
            "--lambda_prior",
            default=0.01,
            type=float,
        )
    
    parser.add_argument(
        "--slicing_dim", # Modificato: da perc_dim (axial/coronal)
        default="spatial",
        type=str,
        # spatial = piano RA-DEC (somma sui canali o slice singola)
        # spectral = piano Posizione-Velocità (slice lungo RA o DEC)
        choices=["spatial", "spectral_ra", "spectral_dec"], 
        help="Direzione lungo cui calcolare la loss percettiva"
    )

    # --- DEGRADAZIONE / CORRUZIONE (Telescope effects) ---
    parser.add_argument(
        "--corruption",
        default="downsample",
        type=str,
        # beam_smear = convoluzione con PSF (Point Spread Function)
        # noise = rumore termico del ricevitore
        choices=["downsample", "mask", "beam_smear", "noise", "None"],
    )
    parser.add_argument(
        "--mask_id",
        default="survey_edge", # es. bordi del rilevatore
        type=str,
    )
    parser.add_argument(
        "--downsample_factor",
        default=4,
        type=int,
        choices=[2, 4, 8, 16],
        help="Fattore di riduzione risoluzione (binning spettrale o spaziale)"
    )
    
    # --- DDIM SAMPLING ---
    parser.add_argument(
        "--ddim_num_timesteps",
        default=10, # 250 è spesso eccessivo per test rapidi
        type=int,
    )
    parser.add_argument(
        "--ddim_eta",
        default=0.0, # 0.0 = Deterministico, 1.0 = DDPM standard
        type=float,
    )
    
    # --- LOSS EXTRA ---
    parser.add_argument(
        "--downsampling_loss",
        action="store_true",
        help="Forza la coerenza tra output HR e input LR"
    )
    parser.add_argument(
        "--mean_latent_vector",
        action="store_true",
        help="Inizia l'inversione dal vettore medio del dataset"
    )
    
    # --- ALTRO ---
    parser.add_argument(
        "--start_steps",
        default=0,
        type=int,
    )
    parser.add_argument(
        "--num_steps",
        default=50,
        type=int,
    )
    parser.add_argument(
        "--prior_every",
        default=20,
        type=int,
    )
    parser.add_argument(
        "--n_samples",
        default=1, # Solitamente 1 datacube alla volta per limiti di memoria
        type=int,
    )

    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        type=str,
    )

    parser.add_argument(
        "--path_to_ddpm_checkpoint",    
        default=".\\data\\trained_models_astro\\ddpm",
        type=str,
        help="Path al checkpoint della DDPM pre-allenata"
    )

    parser.add_argument(
        "--path_to_latent_ddpm",    
        default=".\\data\\trained_models_astro\\results.pth",
        type=str,
        help="Path al checkpoint della DDPM pre-allenata"
    )

    parser.add_argument(
        "--output_dir_BRGM_ddim",    
        default=".\\data\\outputs\\BRGM_ddim",
        type=str,
        help="Path per salvare i risultati dell'inversione"
    )

    parser.add_argument(
        "--output_dir_BRGM_decoder",    
        default=".\\data\\outputs\\decoder",
        type=str,
        help="Path per salvare i risultati dell'inversione"
    )

    parser.add_argument(
        "--vae_path_BRGM",    
        default="/leonardo_scratch/large/userexternal/gvitanza/InverseSR/data/trained_models_astro/vae_decoder_train_100ep_z3_local_1e-4_newloss/VAE_full",
        type=str,
        help="Path per caricare il modello VAE pre-allenato"
    )

    parser.add_argument(
        "--ddpm_path_BRGM",    
        default="/leonardo_scratch/large/userexternal/gvitanza/InverseSR/data/trained_models_astro/ddpm_cross_attn_100_2_z3_local/ddpm_final_model",
        type=str,
        help="Path per caricare il modello DDPM pre-allenato"
    )

    parser.add_argument(
        "--decoder_path_BRGM",    
        default="/leonardo_scratch/large/userexternal/gvitanza/InverseSR/data/trained_models_astro/vae_decoder_train_100ep_z3_local_1e-4_newloss/Decoder_only",
        type=str,
        help="Path per caricare il modello VAE pre-allenato (decoder)"
    )

    parser.add_argument("--cond_key", type=str, default=None)

