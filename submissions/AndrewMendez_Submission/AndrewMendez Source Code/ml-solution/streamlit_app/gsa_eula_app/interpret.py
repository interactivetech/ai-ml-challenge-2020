# import matplotlib.pyplot as plt
# import pandas as pd
import torch
import streamlit as st
import pandas as pd
# Preliminaries

# from torchtext.data import Field, TabularDataset, BucketIterator, Iterator

# Models

# import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer, BertForSequenceClassification

# Training

import torch.optim as optim

# Evaluation

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, brier_score_loss

import seaborn as sns

from captum.attr import visualization as viz
from captum.attr import IntegratedGradients, LayerConductance, LayerIntegratedGradients
from captum.attr import configure_interpretable_embedding_layer, remove_interpretable_embedding_layer
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from time import time
class BERT(nn.Module):

    def __init__(self):
        super(BERT, self).__init__()

        options_name = "bert-base-uncased"
        self.encoder = BertForSequenceClassification.from_pretrained(options_name)

    def forward(self, text, label):
        loss, text_fea = self.encoder(text, labels=label)[:2]

        return loss, text_fea
def load_model(path,device):
    model = BERT().to(device)
    load_checkpoint(path, model,device)
    model.to(device)
    model.eval()
    model.zero_grad()
    return model

def construct_input_ref_pair(text, tokenizer,device, ref_token_id, sep_token_id, cls_token_id):

    text_ids = tokenizer.encode(text, add_special_tokens=False)
    # construct input token ids
    input_ids = [cls_token_id] + text_ids + [sep_token_id]
 
    # construct reference token ids 
    ref_input_ids = [cls_token_id] + [ref_token_id] * len(text_ids) + [sep_token_id]

    return torch.tensor([input_ids], device=device), torch.tensor([ref_input_ids], device=device), len(text_ids)

def construct_input_ref_token_type_pair(input_ids,device, sep_ind=0):
    seq_len = input_ids.size(1)
    token_type_ids = torch.tensor([[0 if i <= sep_ind else 1 for i in range(seq_len)]], device=device)
    ref_token_type_ids = torch.zeros_like(token_type_ids, device=device)# * -1
    return token_type_ids, ref_token_type_ids

def construct_input_ref_pos_id_pair(input_ids,device):
    seq_length = input_ids.size(1)
    position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
    # we could potentially also use random permutation with `torch.randperm(seq_length, device=device)`
    ref_position_ids = torch.zeros(seq_length, dtype=torch.long, device=device)

    position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
    ref_position_ids = ref_position_ids.unsqueeze(0).expand_as(input_ids)
    return position_ids, ref_position_ids
    
def construct_attention_mask(input_ids):
    return torch.ones_like(input_ids)



def save_act(module, inp, out):
  #global saved_act
  #saved_act = out
  return saved_act

def summarize_attributions(attributions):
    attributions = attributions.sum(dim=-1).squeeze(0)
    attributions = attributions / torch.norm(attributions)
    return attributions

def get_topk_attributed_tokens(attrs,all_tokens, k=5):
    values, indices = torch.topk(attrs, k)
    top_tokens = [all_tokens[idx] for idx in indices]
    return top_tokens, values, indices

def load_checkpoint(load_path, model,device):
    
    if load_path==None:
        return
    
    state_dict = torch.load(load_path, map_location=device)

def interpret_main(text,label):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = load_model('/Users/andrewmendez1/Documents/ai-ml-challenge-2020/data/Finetune BERT oversampling 8_16_2020/Model_1_4_0/model.pt',device)
    def predict(inputs):
        #print('model(inputs): ', model(inputs))
        return model.encoder(inputs)[0]
    def custom_forward(inputs):
        preds = predict(inputs)
        return torch.softmax(preds, dim = 1)[:, 0] 
    # load tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    ref_token_id = tokenizer.pad_token_id # A token used for generating token reference
    sep_token_id = tokenizer.sep_token_id # A token used as a separator between question and text and it is also added to the end of the text.
    cls_token_id = tokenizer.cls_token_id # A token used for prepending to the concatenated question-text word sequence
    hook = model.encoder.bert.embeddings.register_forward_hook(save_act)
    hook.remove()


    input_ids, ref_input_ids, sep_id = construct_input_ref_pair(text,tokenizer,device, ref_token_id, sep_token_id, cls_token_id)
    token_type_ids, ref_token_type_ids = construct_input_ref_token_type_pair(input_ids, device,sep_id)
    position_ids, ref_position_ids = construct_input_ref_pos_id_pair(input_ids,device)
    attention_mask = construct_attention_mask(input_ids)

    # text = "the exclusion of implied warranties is not permitted by some the above exclusion may not apply to"# label 0

    lig = LayerIntegratedGradients(custom_forward, model.encoder.bert.embeddings)
    # attributions_main, delta_main = lig.attribute(inputs=input_ids,baselines=ref_input_ids,return_convergence_delta=True,n_steps=30)
    t0 = time()
    attributions, delta = lig.attribute(inputs=input_ids,
                                    baselines=ref_input_ids,
                                    # n_steps=7000,
                                    # internal_batch_size=5,
                                    return_convergence_delta=True,
                                    n_steps=300)
    st.write("Time to complete interpretation: {} seconds".format(time()-t0))
    # print("Time in {} minutes".format( (time()-t0)/60 ))
    attributions_sum = summarize_attributions(attributions)

    all_tokens = tokenizer.convert_ids_to_tokens(input_ids[0].detach().tolist())
    top_tokens, values, indicies = get_topk_attributed_tokens(attributions_sum,all_tokens,k=7)
    st.subheader("Top Tokens that the Model decided Unacceptability")
    import numpy as np
    plt.figure(figsize=(12,6))
    x_pos = np.arange(len(values))
    plt.bar(x_pos,values.detach().numpy(), align='center')
    plt.xticks(x_pos, top_tokens, wrap=True)
    plt.xlabel("Tokens")
    plt.title("Top 5 Tokens that made the model classify clause as unacceptable")
    st.pyplot()

    st.subheader("Detailed Table showing Attribution Score to each word in clause")
    st.write(" ")
    st.write("Positive Attributions mean that the words/tokens were \"positively\" attributed to the models's prediction.")
    st.write("Negative Attributions mean that the words/tokens were \"negatively\" attributed to the models's prediction.")

    # res = ['{}({}) {:.3f}'.format(token, str(i),attributions_sum[i]) for i, token in enumerate(all_tokens)]
    df = pd.DataFrame({'Words':all_tokens,'Attributions':attributions_sum.detach().numpy()})
    st.table(df)
    score = predict(input_ids)
    score_vis = viz.VisualizationDataRecord(attributions_sum,
                                            torch.softmax(score, dim = 1)[0][0],
                                            torch.argmax(torch.softmax(score, dim = 1)[0]),
                                            label,
                                            text,
                                            attributions_sum.sum(),       
                                            all_tokens,
                                            delta)
    print('\033[1m', 'Visualization For Score', '\033[0m')
    # from IPython.display import display, HTML, Image
    # viz.visualize_text([score_vis])
    # st.write(display(Image(viz.visualize_text([score_vis])) ) )

    # open('output.png', 'wb').write(im.data)
    # st.pyplot()



# text= "this license shall be effective until company in its sole and absolute at any time and for any or no disable the or suspend or terminate this license and the rights afforded to you with or without prior notice or other action by upon the termination of this you shall cease all use of the app and uninstall the company will not be liable to you or any third party for or damages of any sort as a result of terminating this license in accordance with its and termination of this license will be without prejudice to any other right or remedy company may now or in the these obligations survive termination of this"
# # label=1
# label = "?"
# main(text,label)