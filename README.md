# DTITR: End-to-End Drut-Target Binding Affinity Prediction with Transformer
<p align="justify"> We propose an end-to-end Transformer-based and DTI-inspired architecture (DTITR) for predicting the logarithmic-transformed quantitative dissociation constant (pKd) of DTI pairs, where self-attention layers are exploited to learn the short and long-term biological and chemical context dependencies between the sequential and structural units of the protein sequences and compound SMILES strings, respectively, and cross-attention layers to exchange information and learn the pharmacological context associated with the interaction space. The architecture makes use of two parallel Transformer-Encoders to compute a contextual embedding of the protein sequences and SMILES strings, and a Cross-Attention Transformer-Encoder block to model the interaction, where the resulting aggregated representation hidden states are concatenated and used as input for a Fully-Connected Feed-Forward Network.</p>


## DTITR Architecture
<p align="center"><img src="/figures/dtitr_arch.png" width="70%" height="70%"/></p>

## Davis Kinase Binding Affinity
### Dataset
- **davis_original_dataset:** original dataset
- **davis_dataset_processed:** dataset processed : prot sequences + rdkit SMILES strings + pkd values
- **deep_features_dataset:** CNN deep representations: protein + SMILES deep representations
### Clusters
- **test_cluster:** independent test set indices
- **train_cluster_X:** train indices 
### Similarity
- **protein_sw_score:** protein Smith-Waterman similarity scores
- **protein_sw_score_norm:** protein Smith-Waterman similarity normalized scores
- **smiles_ecfp6_tanimoto_sim:** SMILES Morgan radius 3 similarity scores

## Dictionaries
- **davis_prot_dictionary**: AA char-integer dictionary
- **davis_smiles_dictionary**: SMILES char-integer dictionary
- **protein_codes_uniprot/subword_units_map_uniprot**: Protein Subwords Dictionary
- **drug_codes_chembl/subword_units_map_chembl**: SMILES Subwords Dictionary

