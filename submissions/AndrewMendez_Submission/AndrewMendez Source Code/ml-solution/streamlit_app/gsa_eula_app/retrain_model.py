# Libraries

import matplotlib.pyplot as plt
import pandas as pd
import torch

# Preliminaries

from torchtext.data import Field, TabularDataset, BucketIterator, Iterator

# Models

import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer, BertForSequenceClassification

# Training

import torch.optim as optim

# Evaluation

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, brier_score_loss

import seaborn as sns
import streamlit as st
import os

class BERT(nn.Module):

    def __init__(self):
        super(BERT, self).__init__()

        options_name = "bert-base-uncased"
        self.encoder = BertForSequenceClassification.from_pretrained(options_name)

    def forward(self, text, label):
        loss, text_fea = self.encoder(text, labels=label)[:2]

        return loss, text_fea

def get_dataset_filenames(dataset_directory):
    filenames = [ os.path.join(dataset_directory,i) for i in os.listdir(dataset_directory)]
    filenames = [i.split("/")[-1] for i in filenames if os.path.isfile(i)]
    return filenames[::-1]
def load_dataset(source_folder,
                 device,
                 tokenizer,
                 MAX_SEQ_LEN=128,
                 BATCH_SIZE=16,
                 name_of_train_dataset='train.csv',
                 name_of_validation_dataset='valid.csv',
                 name_of_test_dataset='test.csv'):

    PAD_INDEX = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    UNK_INDEX = tokenizer.convert_tokens_to_ids(tokenizer.unk_token)

    # Fields

    label_field = Field(sequential=False,
                        use_vocab=False,
                        batch_first=True,
                        dtype=torch.float)
    
    text_field = Field(use_vocab=False,
                        tokenize=tokenizer.encode,
                        lower=False,
                        include_lengths=False,
                        batch_first=True,
                        fix_length=MAX_SEQ_LEN,
                        pad_token=PAD_INDEX,
                        unk_token=UNK_INDEX)
    # note, the fields must be in the same order as the input csv columns
    fields = [('clause_text', text_field),('label', label_field)]

    # TabularDataset

    train, valid, test = TabularDataset.splits(path=source_folder,
                                               train=name_of_train_dataset,
                                               validation=name_of_validation_dataset,
                                               test=name_of_test_dataset,
                                               format='CSV',
                                               fields=fields,
                                               skip_header=True)

    # Iterators

    train_iter = BucketIterator(train, batch_size=BATCH_SIZE, sort_key=lambda x: len(x.clause_text),
                                device=device, train=True, sort=True, sort_within_batch=True)
    valid_iter = BucketIterator(valid, batch_size=BATCH_SIZE, sort_key=lambda x: len(x.clause_text),
                                device=device, train=True, sort=True, sort_within_batch=True)
    test_iter = Iterator(test, batch_size=BATCH_SIZE, device=device, train=False, shuffle=False, sort=False)
    return train_iter,valid_iter,test_iter

def save_checkpoint(save_path, model, valid_loss):

    if save_path == None:
        return
    
    state_dict = {'model_state_dict': model.state_dict(),
                  'valid_loss': valid_loss}
    
    torch.save(state_dict, save_path)
    print(f'Model saved to ==> {save_path}')
    st.write(f'Model saved to ==> {save_path}')

def load_checkpoint(load_path, model,device):
    
    if load_path==None:
        return
    
    state_dict = torch.load(load_path, map_location=device)
    print(f'Model loaded from <== {load_path}')
    st.write(f'Model loaded from <== {load_path}')
    model.load_state_dict(state_dict['model_state_dict'])
    return state_dict['valid_loss']


def save_metrics(save_path, train_loss_list, valid_loss_list, global_steps_list):

    if save_path == None:
        return
    
    state_dict = {'train_loss_list': train_loss_list,
                  'valid_loss_list': valid_loss_list,
                  'global_steps_list': global_steps_list}
    
    torch.save(state_dict, save_path)
    print(f'Model saved to ==> {save_path}')
def load_metrics(load_path,device):
    if load_path==None:
        return
    
    state_dict = torch.load(load_path, map_location=device)
    print(f'Model loaded from <== {load_path}')
    st.write(f'Model loaded from <== {load_path}')
    
    return state_dict['train_loss_list'], state_dict['valid_loss_list'], state_dict['global_steps_list']
def train(model,
          optimizer,
          device,
          train_loader,
          valid_loader,
          criterion = nn.BCELoss(),
          num_epochs = 1,
          eval_every=1,
          file_path='.',
          best_valid_loss = float("Inf")):
    
    # initialize running values
    running_loss = 0.0
    valid_running_loss = 0.0
    global_step = 0
    train_loss_list = []
    valid_loss_list = []
    global_steps_list = []

    # training loop
    model.train()
    for epoch in range(num_epochs):
        for (text, label), _ in train_loader:
            label = label.type(torch.LongTensor)           
            label = label.to(device)
            text = text.type(torch.LongTensor)  
            text = text.to(device)
            output = model(text, label)
            loss, _ = output

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # update running values
            running_loss += loss.item()
            global_step += 1

            # evaluation step
            if global_step % eval_every == 0:
                model.eval()
                with torch.no_grad():                    

                    # validation loop
                    for (text,label), _ in valid_loader:
                        label = label.type(torch.LongTensor)           
                        label = label.to(device)
                        text = text.type(torch.LongTensor)  
                        text = text.to(device)
                        output = model(text, label)
                        loss, _ = output
                        
                        valid_running_loss += loss.item()

                # evaluation
                average_train_loss = running_loss / eval_every
                average_valid_loss = valid_running_loss / len(valid_loader)
                train_loss_list.append(average_train_loss)
                valid_loss_list.append(average_valid_loss)
                global_steps_list.append(global_step)

                # resetting running values
                running_loss = 0.0                
                valid_running_loss = 0.0
                model.train()

                # print progress
                print('Epoch [{}/{}], Step [{}/{}], Train Loss: {:.4f}, Valid Loss: {:.4f}'
                      .format(epoch+1, num_epochs, global_step, num_epochs*len(train_loader),
                              average_train_loss, average_valid_loss))
                st.write('Epoch [{}/{}], Step [{}/{}], Train Loss: {:.4f}, Valid Loss: {:.4f}'
                      .format(epoch+1, num_epochs, global_step, num_epochs*len(train_loader),
                              average_train_loss, average_valid_loss))
                
                # checkpoint
                if best_valid_loss > average_valid_loss:
                    best_valid_loss = average_valid_loss
                    save_checkpoint(file_path + '/' + 'model.pt', model, best_valid_loss)
                    save_metrics(file_path + '/' + 'metrics.pt', train_loss_list, valid_loss_list, global_steps_list)
    
    save_metrics(file_path + '/' + 'metrics.pt', train_loss_list, valid_loss_list, global_steps_list)
    print('Finished Training!')

def plot_train_and_valid_losses(destination_folder,device):
    train_loss_list, valid_loss_list, global_steps_list = load_metrics(destination_folder + '/metrics.pt',device)
    plt.plot(global_steps_list, train_loss_list, label='Train')
    plt.plot(global_steps_list, valid_loss_list, label='Valid')
    plt.xlabel('Global Steps')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    st.pyplot()

def evaluate(model, test_loader,device):
    y_pred = []
    y_true = []
    output_probs = []
    model.eval()
    with torch.no_grad():
        for ( text, label), _ in test_loader:

                label = label.type(torch.LongTensor)           
                label = label.to(device)
                text = text.type(torch.LongTensor)  
                text = text.to(device)
                output = model(text, label)

                _, output = output
                output_probs.extend(F.softmax(output, dim=0).tolist())
                y_pred.extend(torch.argmax(output, 1).tolist())
                y_true.extend(label.tolist())
    
    # Brier score measures the mean squared difference between
    # (1) the predicted probability assigned to the possible outcomes for item i,
    # and (2) the actual outcome. Therefore, the lower the Brier score is for a set
    # of predictions, the better the predictions are calibrated.
    clf_score = brier_score_loss(y_true, [output_probs[i][y_pred[i]] for i in range(len(output_probs))], pos_label=0)
    print("Brier Score: %1.3f" % clf_score)
    st.write("Brier Score: %1.3f" % clf_score)
    print('Classification Report:')
    st.markdown(classification_report(y_true, y_pred, labels=[1,0], digits=4))
    
    cm = confusion_matrix(y_true, y_pred, labels=[1,0])
    ax= plt.subplot()
    sns.heatmap(cm, annot=True, ax = ax, cmap='Blues', fmt="d")

    ax.set_title('Confusion Matrix')

    ax.set_xlabel('Predicted Labels')
    ax.set_ylabel('True Labels')

    ax.xaxis.set_ticklabels(['Good', 'Bad'])
    ax.yaxis.set_ticklabels(['Good', 'Bad'])
    st.pyplot()

def retrain_model_main(source_folder,destination_folder,num_epochs=1,batch_size=2):
        with st.spinner('Training Model...'):
        # dataset_directory = '/Users/andrewmendez1/Documents/ai-ml-challenge-2020/data/dummy_retrain_data'
            filenames = get_dataset_filenames(source_folder)

            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            print(device)
            st.write("Loading Model...")
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            train_iter,valid_iter,test_iter = load_dataset(source_folder,device,tokenizer,128,batch_size,filenames[0],filenames[1],filenames[2])


            model = BERT().to(device)
            optimizer = optim.Adam(model.parameters(), lr=2e-5)
            eval_every = len(train_iter) // 2
            train(model=model, optimizer=optimizer,
                train_loader=train_iter,
                valid_loader=valid_iter,
                device=device,
                file_path=destination_folder,
                num_epochs=num_epochs,
                eval_every=eval_every)

            plot_train_and_valid_losses(destination_folder,device)

            best_model = BERT().to(device)
            load_checkpoint(destination_folder + '/model.pt', best_model,device)
            evaluate(best_model, test_iter,device)
    

# source_folder = '/Users/andrewmendez1/Documents/ai-ml-challenge-2020/data/dummy_retrain_data'
# destination_folder = '/Users/andrewmendez1/Documents/ai-ml-challenge-2020/data/dummy_retrain_data/model_result'