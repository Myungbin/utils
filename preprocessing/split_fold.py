# !pip install split-folders
"""
Before
DataFolder
├─ Malware
│  └─ PE file image set
└─ Normal
   └─ PE file image set

After
DataFolder
├─ train
│  ├─ Malware
│  └─ Normal
├─ val
│  ├─ Malware
└─ └─ Normal
"""
import splitfolders

splitfolders.ratio(r"C:\Project\Data\clf", r"C:\Project\Data\clf_split", seed=1103, ratio=(0.7, 0.3))
"""
splitfolders.ratio(file_path, result_path, ratio=(train, val ration))
"""