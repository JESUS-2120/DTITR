from transformer_encoder import *
from cross_attention_transformer_encoder import *
from embedding_layer import *
from layers_utils import *
from output_block import *
from dataset_builder_util import *
import itertools
import tensorflow_addons as tfa
from argument_parser import *
import gc
from plot_utils import *
from utils import *

#Use gpu
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

def build_dtitr_model(FLAGS, prot_trans_depth, smiles_trans_depth, cross_attn_depth,
                      prot_trans_heads, smiles_trans_heads, cross_attn_heads,
                      prot_parameter_sharing, prot_dim_k,
                      prot_d_ff, smiles_d_ff, d_model, dropout_rate, dense_atv_fun,
                      out_mlp_depth, out_mlp_units, optimizer_fn):

    if FLAGS.bpe_option[0]:
        prot_input = tf.keras.Input(shape=(FLAGS.protein_bpe_len + 1,), dtype=tf.int32, name='protein_input')
        prot_mask = attn_pad_mask()(prot_input)
        encode_prot = EmbeddingLayer(FLAGS.protein_dict_bpe_len + 2, d_model,  # FLAGS.protein_bpe_len+1,
                                     dropout_rate, FLAGS.pos_enc_option)(prot_input)

    else:
        prot_input = tf.keras.Input(shape=(FLAGS.protein_len + 1,), dtype=tf.int32, name='protein_input')
        prot_mask = attn_pad_mask()(prot_input)
        encode_prot = EmbeddingLayer(FLAGS.protein_dict_len + 2, d_model,  # FLAGS.protein_len+1,
                                     dropout_rate, FLAGS.pos_enc_option)(prot_input)

    encode_prot, _ = Encoder(d_model, prot_trans_depth, prot_trans_heads, prot_d_ff, dense_atv_fun,
                             dropout_rate, prot_dim_k, prot_parameter_sharing,
                             FLAGS.prot_full_attn,
                             FLAGS.return_intermediate, name='encoder_prot')(encode_prot, prot_mask)

    if FLAGS.bpe_option[1]:
        smiles_input = tf.keras.Input(shape=(FLAGS.smiles_bpe_len + 1,), dtype=tf.int32, name='smiles_input')
        smiles_mask = attn_pad_mask()(smiles_input)
        encode_smiles = EmbeddingLayer(FLAGS.smiles_dict_bpe_len + 2, d_model,  # FLAGS.smiles_bpe_len+1,
                                       dropout_rate, FLAGS.pos_enc_option)(smiles_input)
    else:
        smiles_input = tf.keras.Input(shape=(FLAGS.smiles_len + 1,), dtype=tf.int32, name='smiles_input')
        smiles_mask = attn_pad_mask()(smiles_input)
        encode_smiles = EmbeddingLayer(FLAGS.smiles_dict_len + 2, d_model,  # FLAGS.smiles_len+1,
                                       dropout_rate, FLAGS.pos_enc_option)(smiles_input)

    encode_smiles, _ = Encoder(d_model, smiles_trans_depth, smiles_trans_heads, smiles_d_ff, dense_atv_fun,
                               dropout_rate, FLAGS.smiles_dim_k, FLAGS.smiles_parameter_sharing,
                               FLAGS.smiles_full_attn, FLAGS.return_intermediate,
                               name='encoder_smiles')(encode_smiles, smiles_mask)

    cross_prot_smiles, _ = CrossAttnBlock(d_model, cross_attn_depth, cross_attn_heads, prot_trans_heads,
                                          smiles_trans_heads, prot_d_ff, smiles_d_ff, dense_atv_fun,
                                          dropout_rate, prot_dim_k, prot_parameter_sharing,
                                          FLAGS.prot_full_attn, FLAGS.smiles_dim_k,
                                          FLAGS.smiles_parameter_sharing, FLAGS.smiles_full_attn,
                                          FLAGS.return_intermediate,
                                          name='cross_attn_block')([encode_prot,
                                                                    encode_smiles],
                                                                   smiles_mask,
                                                                   prot_mask)

    out = OutputMLP(out_mlp_depth, out_mlp_units, dense_atv_fun,
                    FLAGS.output_atv_fun, dropout_rate, name='output_block')(cross_prot_smiles)


    dtitr_model = tf.keras.Model(inputs=[prot_input, smiles_input], outputs=out, name='dtitr')

    dtitr_model.compile(optimizer=optimizer_fn, loss=FLAGS.loss_function,
                        metrics=[tf.keras.metrics.RootMeanSquaredError(), c_index])

    return dtitr_model



FLAGS = argparser()
FLAGS.log_dir = os.getcwd() + '/logs/' + time.strftime("%d_%m_%y_%H_%M", time.gmtime()) + "/"
FLAGS.checkpoint_path = os.getcwd() + '/checkpoints/' + time.strftime("%d_%m_%y_%H_%M", time.gmtime()) + "/"
FLAGS.data_path = {'data': '../data/Data_TaP.csv',
                   'prot_dic': '../dictionary/davis_prot_dictionary.txt',
                   'smiles_dic': '../dictionary/davis_smiles_dictionary.txt',
                   'prot_bpe': ['../dictionary/protein_codes_uniprot.txt',
                                    '../dictionary/subword_units_map_uniprot.csv'],
                   'smiles_bpe': ['../dictionary/drug_codes_chembl.txt',
                                      '../dictionary/subword_units_map_chembl.csv']}


#Prepare data
protein_data, smiles_data = dataset_builder(FLAGS.data_path).transform_dataset(FLAGS.bpe_option[0],FLAGS.bpe_option[1],'Sequence','SMILES','Kd',FLAGS.protein_bpe_len,FLAGS.protein_len,FLAGS.smiles_bpe_len,FLAGS.smiles_len)
                             
if FLAGS.bpe_option[0] == True:
    protein_data = add_reg_token(protein_data, FLAGS.protein_dict_bpe_len)
else:
    protein_data = add_reg_token(protein_data, FLAGS.protein_dict_len)

if FLAGS.bpe_option[1] == True:
    smiles_data = add_reg_token(smiles_data, FLAGS.smiles_dict_bpe_len)
else:
    smiles_data = add_reg_token(smiles_data, FLAGS.smiles_dict_len)

optimizer_fun = tfa.optimizers.RectifiedAdam(learning_rate=1e-04, beta_1=0.9,
                                                 beta_2=0.999, epsilon=1e-08,
                                                 weight_decay=1e-05)
                                                 
#Load model and predict
dtitr_model = build_dtitr_model(FLAGS, 3, 3, 1, 4, 4, 4, '', '', 512, 512, 128, 0.1, 'gelu', 3, [512, 512, 512],optimizer_fun)

dtitr_model.load_weights('../model/dtitr_model/')

prediccion = dtitr_model.predict([protein_data, smiles_data])

file_n = open("../Prediction_1.txt","w")
accession = dataset_builder(FLAGS.data_path).get_data()[0]['Acession Number']

for i in range(len(accession)):
  file_n.write(str(accession[i])+":"+str(prediccion[i])+"\n")
