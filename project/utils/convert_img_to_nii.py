import nibabel as nib

if __name__ == "__main__":
    file_path = r"C:\Users\16446\Documents\GitHub\Medical-Image-Reconstruction\data\OASIS\oasis-1\disc1\OAS1_0002_MR1\PROCESSED\MPRAGE\T88_111\OAS1_0002_MR1_mpr_n4_anon_111_t88_gfc.img"
    img = nib.load(file_path)
    nib.save(img, file_path.replace(".img", ".nii"))
