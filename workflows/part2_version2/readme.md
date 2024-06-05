
# Retention time predictor 

- Courtesy of Dan Pasin (nps-rt-main) and some updated libraries 

> Note that for training the retention time predictor, we only want to use retention times that have been confirmed at PTC. Please use Training_Database_with_fbs_rt.csv column Y (PTC Confirmed RT) only. You can ignore column U (Retention Time)

## Fields in ```Prediction Data.csv``` (n=2329)
| Compound | DrugClass| InChIKey| InChIKeyShort | SMILES | logD | logP | nO | nC | 
| :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- |
| (Iso)butyryl-F-fentanyl N-benzyl analogue  |Opioids  | XNQGKYHSTDKIKG-UHFFFAOYSA-N  | XNQGKYHSTDKIKG  |  (C1=CC=CC=C1)N1CCC(CC1)N(C(C(C)C)=O)C1=CC=C(C=C1)F  | 0.717182  | 4.4793  | 1  | 22 | 


## ```Modeling Data``` (n=4770)

| Lab | Compound | RT | DrugClass| InChIKey| InChIKeyShort | SMILES | logD | logP | nO | nC | 
| :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- |
| Aarhus |1B-LSD |5.55 |Indolalkylamines |SVRFNPSJPIDUBC-DYESRHJHSA-N | SVRFNPSJPIDUBC  | C(CCC)(=O)N1C=C2C[C@H]3N(C[C@@H] | C=C3C=3C=CC=C1C32)C(=O)N(CC)CC)C  |-0.70648  | 3.8197 | 2 |24 | 
