This is a repo to re-verify the BraTS-Mets 2023 results. This is forked from the original repo.

# To reproduce

1. Prepare the data by running `datasetup.sh`
2. Run `python analysis` to reproduce the table 6 in the paper.

Note: change `from metrics import get_LesionWiseResults` to `from metric_cupy import get_LesionWiseResults` to use a gpu speedup. Original code is only numpy based.

# Data Hierarchy

```bash
 └── dataset
     ├── ASNR-MICCAI-BraTS2023-MET-Challenge-TestingData
     │   ├── BraTS-MET-00001-000
     │   │   ├── BraTS-MET-00001-000-seg.nii.gz
     │   │   ├── BraTS-MET-00001-000-t1c.nii.gz
     │   │   ├── BraTS-MET-00001-000-t1n.nii.gz
     │   │   ├── BraTS-MET-00001-000-t2f.nii.gz
     │   │   └── BraTS-MET-00001-000-t2w.nii.gz
     │   ├── BraTS-MET-00018-000
     │   │   ├── BraTS-MET-00018-000-seg.nii.gz
     │   │   ├── BraTS-MET-00018-000-t1c.nii.gz
     │   │   ├── BraTS-MET-00018-000-t1n.nii.gz
     │   │   ├── BraTS-MET-00018-000-t2f.nii.gz
     │   │   └── BraTS-MET-00018-000-t2w.nii.gz
     │   ├── BraTS-MET-00137-000
     │   │   ├── BraTS-MET-00137-000-seg.nii.gz
     │   │   ├── BraTS-MET-00137-000-t1c.nii.gz
     │   │   ├── BraTS-MET-00137-000-t1n.nii.gz
     │   │   ├── BraTS-MET-00137-000-t2f.nii.gz
     │   │   └── BraTS-MET-00137-000-t2w.nii.gz
     │   ├── BraTS-MET-00144-000
     │   │   ├── BraTS-MET-00144-000-seg.nii.gz
     |   |   |... And so on
     │   ├── BratSMets_PredictedSegs.zip
     │   └── task4_NVAUTO_9740011_JR.xlsx
     ├── BratSMets_PredictedSegs
     │   ├── PredictedSegs
     │   │   ├── CNMC_PMI2023
     │   │   │   ├── BraTS-MET-00001-000.nii.gz
     │   │   │   ├── BraTS-MET-00018-000.nii.gz
     │   │   │   ├── BraTS-MET-00137-000.nii.gz
     │   │   │   ├── BraTS-MET-00144-000.nii.gz
     │   │   │   ├── BraTS-MET-00147-000.nii.gz
     │   │   │   ├── ...
     │   │   ├── MIA_SINTEF
     │   │   │   ├── BraTS-MET-00001-000.nii.gz
     │   │   │   ├── BraTS-MET-00018-000.nii.gz
     │   │   │   ├── BraTS-MET-00137-000.nii.gz
     │   │   │   ├── BraTS-MET-00144-000.nii.gz
     │   │   │   ├── BraTS-MET-00147-000.nii.gz
     │   │   │   ├── ...
     │   │   ├── NVAUTO
     │   │   │   ├── BraTS-MET-00001-000.nii.gz
     │   │   │   ├── BraTS-MET-00018-000.nii.gz
     │   │   │   ├── BraTS-MET-00137-000.nii.gz
     │   │   │   ├── BraTS-MET-00144-000.nii.gz
     │   │   │   ├── BraTS-MET-00147-000.nii.gz
     │   │   │   ├── ...
     │   │   ├── S_Y
     │   │   │   ├── BraTS-MET-00001-000.nii.gz
     │   │   │   ├── BraTS-MET-00018-000.nii.gz
     │   │   │   ├── BraTS-MET-00137-000.nii.gz
     │   │   │   ├── BraTS-MET-00144-000.nii.gz
     │   │   │   ├── BraTS-MET-00147-000.nii.gz
     │   │   │   ├── ...
     │   │   ├── blackbean
     │   │   │   ├── BraTS-MET-00001-000.nii.gz
     │   │   │   ├── BraTS-MET-00018-000.nii.gz
     │   │   │   ├── BraTS-MET-00137-000.nii.gz
     │   │   │   ├── BraTS-MET-00144-000.nii.gz
     │   │   │   ├── BraTS-MET-00147-000.nii.gz
     │   │   │   ├── ...
     │   │   └── i_sahajmistry
     │   │       ├── BraTS-MET-00001-000.nii.gz
     │   │       ├── BraTS-MET-00018-000.nii.gz
     │   │       ├── BraTS-MET-00137-000.nii.gz
     │   │       ├── BraTS-MET-00144-000.nii.gz
     │   │       ├── BraTS-MET-00147-000.nii.gz
     │   │       ├── ...
     │   └── __MACOSX
     │       └── PredictedSegs
     │           ├── CNMC_PMI2023
     │           ├── MIA_SINTEF
     │           ├── NVAUTO
     │           ├── S_Y
     │           ├── blackbean
     │           └── i_sahajmistry
     ├── BratSMets_PredictedSegs.zip
     └── task4_NVAUTO_9740011_JR.xlsx
```