import warnings
warnings.filterwarnings('ignore')

import random

# custom data format gate keeper
from utils.dataloader import create_dataloader

# encoder,decoder ,loss:contrasive ,mae,
from model.model import GraphEncoder, GIN_decoder#, infoNCE_loss, cca_loss, cross_contrastive_loss
from model.loss import contrastive_loss_cell, mae_loss_cell, infoNCE_loss

# torch_metric + torch_geometric -> PyG-based.
import torch
import torch.optim as optim
import torch_geometric.transforms as T
import torch.nn.functional as F
import torch.nn as nn
import torch_geometric.nn as pyg_nn
from torch_geometric.nn.pool import global_add_pool
from torchinfo import summary


from sklearn.model_selection import train_test_split
from tqdm import tqdm
from glob import glob
from argparse import ArgumentParser

# about memory. dataset is big
import math
import os
import psutil
import sys
import numpy as np
import gc
import time
gc.enable()
import logging

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def print_ram_usage():
    process = psutil.Process()  # Get current process
    memory_info = process.memory_info()  # Get memory information
    ram_usage_mb = memory_info.rss / (1024 * 1024 * 1024)  # Convert bytes to MB
    print(f"RAM Usage: {ram_usage_mb:.2f} GB")

'''
--data_dir           default data/pretraining/
--pe_dim             positional encoding dim (128)
--init_dim, --hidden_dim, --output_dim
--blending, --cross_message_passing, --anchor_pe, --pe (all flags)
--num_layers 10
--num_heads 8
--batch_size 128
--lr 1e-3
--wd 3e-3
--num_epochs 20
--gpu 0
'''
parser = ArgumentParser(description="SCGFM")
parser.add_argument('--data_dir', type=str, default = 'data/pretraining/', help="Directory where the raw data is stored")
parser.add_argument('--pe_dim', type=int, default= 128, help="Dimension of the positional encodings")
parser.add_argument('--init_dim', type=int, default= 128, help="Hidden dim for the MLP")
parser.add_argument('--hidden_dim', type=int, default= 128, help="Hidden dim for the MLP")
parser.add_argument('--output_dim', type=int, default= 128, help="Output dim for the MLP")
parser.add_argument('--blending', action='store_true')
parser.add_argument('--cross_message_passing', action='store_true')
parser.add_argument('--anchor_pe', action='store_true') # not used
parser.add_argument('--num_layers', type=int, default= 10, help="Number of MLP layers")
parser.add_argument('--pe', action='store_true') 
parser.add_argument('--num_heads', type=int, default= 8, help="Number of transformer heads")
parser.add_argument('--batch_size', type=int, default= 128, help="Batch size")
parser.add_argument('--graph_idx', type=int, default= 0, help="Batch size")
parser.add_argument('--lr', type=float, default= 1e-3, help="Learnign Rate")
parser.add_argument('--wd', type=float, default= 3e-3, help="Weight decay")
parser.add_argument('--num_epochs', type=int, default= 20, help="Number of epochs")
parser.add_argument('--gpu', type=int, default= 0, help="GPU index")
parser.add_argument('--wandb', action='store_true', help="Enable Weights & Biases logging")
parser.add_argument('--wandb_project', type=str, default='HEIST', help="W&B project name")
parser.add_argument('--wandb_entity', type=str, default=None, help="W&B entity (team/username), optional")

def initialize_weights(layer):
    # Handle linear layers with Xavier initialization
    if isinstance(layer, nn.Linear):
        nn.init.xavier_uniform_(layer.weight)
        if layer.bias is not None:
            nn.init.constant_(layer.bias, 0)

    # Handle TransformerConv layers
    elif isinstance(layer, pyg_nn.TransformerConv):
        # Loop over the named parameters and initialize them if possible
        for name, param in layer.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)

    # Handle Conv2d layers (if applicable)
    elif isinstance(layer, nn.Conv2d):
        nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
        if layer.bias is not None:
            nn.init.constant_(layer.bias, 0)


def validate(model, graphs_list, val_idx):
    total_loss = 0
    with torch.no_grad():
        for graph_idx in val_idx:
            graphs = torch.load(graphs_list[graph_idx], weights_only = False)
            try:
                dataloader = create_dataloader(graphs, args.batch_size, False)
            except KeyError:
                continue

            for high_level_subgraph, low_level_batch, batch_idx in tqdm(dataloader):
                # Move data to device only when needed
                high_level_subgraph = high_level_subgraph.to(args.device) # batch_size \times 2
                low_level_batch = low_level_batch.to(args.device) # batch_size * num_genes \times 1
                low_level_batch.batch_idx = batch_idx.to(args.device)
                low_level_batch.X = low_level_batch.X * 100
            
                # Keep validation objective aligned with training objective.
                high_mask = 1 - torch.bernoulli(torch.ones(high_level_subgraph.num_nodes, 1)*0.1).long().to(args.device)
                low_mask = 1 - torch.bernoulli(torch.ones(low_level_batch.num_nodes, 1)*0.1).long().to(args.device)

                # high_emb, low_emb, aux_loss = model(high_level_subgraph, low_level_batch, high_mask, low_mask)
                high_emb, low_emb = model(high_level_subgraph, low_level_batch, high_mask, low_mask)
                
                contrastive_loss = contrastive_loss_cell(low_level_batch.cell_type, high_emb, low_level_batch, low_emb, 10)
                _high_emb = high_emb * high_mask
                _low_emb = low_emb * low_mask

                decoded_high, decoded_low,alpha = decoder(_high_emb, high_level_subgraph, _low_emb, low_level_batch)
                high_denom = high_mask.sum().float().clamp(min=1.0)
                low_denom = low_mask.sum().float().clamp(min=1.0)
                if (1 - high_mask).sum() > 0:
                    recon_loss = F.mse_loss(decoded_high*(1-high_mask), high_level_subgraph.X.float()*(1-high_mask), reduction='sum')/high_denom + F.mse_loss(decoded_low*(1-low_mask), low_level_batch.X.float()*(1-low_mask), reduction='sum')/low_denom
                else:
                    recon_loss = F.mse_loss(decoded_low*(1-low_mask), low_level_batch.X.float()*(1-low_mask), reduction='sum')/low_denom

                loss = F.sigmoid(decoder.alpha) * contrastive_loss + (1 - F.sigmoid(decoder.alpha)) * recon_loss

                if math.isfinite(loss.item()):
                    total_loss += loss.item()
                del(high_level_subgraph, low_level_batch, loss, high_emb, low_emb, contrastive_loss, recon_loss, high_mask, low_mask, _high_emb, _low_emb)
                torch.cuda.empty_cache()
                gc.collect()
            del(dataloader)
            gc.collect()
    return total_loss
def _get_device(gpu):
    """Use CPU if CUDA device is incompatible or gpu=-1."""
    if gpu == -1:
        return 'cpu'
    if not torch.cuda.is_available():
        return 'cpu'
    try:
        torch.zeros(1, device=f'cuda:{gpu}')
        return f'cuda:{gpu}'
    except RuntimeError as e:
        if 'no kernel image' in str(e) or 'CUDA' in str(e):
            import warnings
            warnings.warn(f"CUDA incompatible ({e}), using CPU. Training will be slower.")
            return 'cpu'
        raise

args = parser.parse_args()
args.device = _get_device(args.gpu)

INPUT_DIM_HIGH = 2
INPUT_DIM_LOW = 1

# the whole procedure :
# encode ->align(contrasive)->reconstruct missing parts*(MAE/MSE)
# ->validate->save best model

if __name__ == '__main__':
    # the folder like: data/pretraining/<sth>/<file>,
    # later loaded with torch.load(...,wei_only-false),so
    # these are serialized PyTorch objects  
    graphs_list = glob(args.data_dir+"*/*") #[torch.load(file, weights_only = False) for file in glob(args.data_dir+"*/*")]
    
    # 80% training 20% validation 
    train_idx, val_idx = train_test_split(np.arange(len(graphs_list)), test_size = 0.2, random_state = 42)
    model_path = f"saved_models/model_6M_attention_anchor_pe_cross_attention_blending_orthogonal_sea_only_moe.pth"
    print(args)

    # GraphEncoder : use PE, do cross-messaging-passing,high attentionish stuff(num_heads),10layers 
    # model = GraphEncoder(args.pe_dim, args.init_dim, args.hidden_dim, args.output_dim, 
    #                        args.num_layers, args.num_heads, args.cross_message_passing, args.pe, args.anchor_pe, args.blending).to(args.device)
    model = GraphEncoder(args.pe_dim, args.init_dim, args.hidden_dim, args.output_dim, 
                            args.num_layers, args.num_heads, args.cross_message_passing, args.pe, args.blending).to(args.device)
    # custom init for linear transformerConv Conv2d. Stable training.
    model.apply(initialize_weights)

    #use GIN to decode ,and it takes the embeddings and reconstruct node features
    decoder = GIN_decoder(args.output_dim, args.output_dim).to(args.device)
    # decoder.apply(initialize_weights)

    # optimizer for both model(encoder) and decoder
    optimizer = optim.AdamW(list(model.parameters())+list(decoder.parameters()), lr=args.lr, weight_decay = args.wd)

    summary(model)
    model.train()

    # Initialize wandb if enabled
    if args.wandb and WANDB_AVAILABLE:
        config_dict = {k: v for k, v in vars(args).items() if k not in ("device",) and isinstance(v, (str, int, float, bool, type(None)))}
        wandb_init_kwargs = {
            "project": args.wandb_project,
            "config": config_dict,
            "name": f"pe{args.pe}_cross{args.cross_message_passing}_blend{args.blending}",
        }
        if args.wandb_entity:
            wandb_init_kwargs["entity"] = args.wandb_entity
        wandb.init(**wandb_init_kwargs)
    elif args.wandb and not WANDB_AVAILABLE:
        print("Warning: wandb requested but not installed. Run: pip install wandb")

    # the training loop : 
    '''
    for epoch in range(num):
        for each training file:
            1. load it
            2. make dataloader
            3. loop over batches
            4. forward 
            5. losses
            6. backward
        run validate
        may save model
    '''
    best_val_loss = 10e9
    for epoch in range(args.num_epochs):
        total_loss = 0
        start_time = time.time()
        for i, graph_idx in enumerate(train_idx):
            graphs = torch.load(graphs_list[graph_idx], weights_only = False)
            try:
                dataloader = create_dataloader(graphs, args.batch_size)
            except KeyError:
                continue
            except RuntimeError:
                continue

            # each batch is a triple:
            # high_level_subgraph: pyG graph on the high level 
            for high_level_subgraph, low_level_batch, batch_idx in tqdm(dataloader):
                optimizer.zero_grad()
                high_level_subgraph = high_level_subgraph.to(args.device) # batch_size \times 2
                
                low_level_batch = low_level_batch.to(args.device) # batch_size * num_genes \times 1
                low_level_batch.batch_idx = batch_idx.to(args.device)
                low_level_batch.X = low_level_batch.X * 100
                # masking , nodes to make it a reconstruction/MAE-style task
                high_mask = 1 - torch.bernoulli(torch.ones(high_level_subgraph.num_nodes, 1)*0.1).long().to(args.device)
                low_mask = 1 - torch.bernoulli(torch.ones(low_level_batch.num_nodes, 1)*0.1).long().to(args.device)
                # with prob 0.1 to drop a node mask =0 
                # with prob 0.9 they keep it   mask =1
                # later reconstruct the dropped part and compute loss ther


                # forward pass
                high_emb, low_emb = model(high_level_subgraph, low_level_batch, high_mask, low_mask)
                
                # losses: compute 3 conceptual things :
                 
                # between constrasive loss between high- and low- views
                contrastive_loss = contrastive_loss_cell(low_level_batch.cell_type, high_emb, low_level_batch, low_emb, 10)
                _high_emb = high_emb * high_mask
                _low_emb = low_emb * low_mask

                # reconstruction loss on the masked nodes
                decoded_high, decoded_low, alpha = decoder(_high_emb, high_level_subgraph, _low_emb, low_level_batch)

                # recon_loss = mae_loss_cell(high_level_subgraph.X, low_level_batch.X, decoded_high, decoded_low, 1 - high_mask, 1 - low_mask)
                high_denom = high_mask.sum().float().clamp(min=1.0)
                low_denom = low_mask.sum().float().clamp(min=1.0)
                if (1 - high_mask).sum() > 0:
                    recon_loss = F.mse_loss(decoded_high*(1-high_mask), high_level_subgraph.X.float()*(1-high_mask), reduction='sum')/high_denom + F.mse_loss(decoded_low*(1-low_mask), low_level_batch.X.float()*(1-low_mask), reduction='sum')/low_denom
                else:
                    recon_loss = F.mse_loss(decoded_low*(1-low_mask), low_level_batch.X.float()*(1-low_mask), reduction='sum')/low_denom

                # orthogonal_loss = 0.1*(_high_emb.T@_high_emb - torch.eye(_high_emb.shape[1]).to(args.device)).square().mean() + 0.1*(_low_emb.T@_low_emb - torch.eye(_low_emb.shape[1]).to(args.device)).square().mean()
                
                # the final loss : learnable mixture
                # the decoder.alpha learns how much to trust contrasive vs reconstruction
                loss = F.sigmoid(decoder.alpha) * contrastive_loss + (1 - F.sigmoid(decoder.alpha)) * recon_loss #+ orthogonal_loss
                if not torch.isfinite(loss):
                    continue  # skip this batch when loss is nan/inf
                total_loss += loss.item()
                # loss = model.forward_contrastive(high_level_subgraph, low_level_batch)
                
                # aggressively free memory
                loss.backward()
                optimizer.step()
                # import pdb; pdb.set_trace()
                del(high_level_subgraph, low_level_batch, loss, high_emb, low_emb, contrastive_loss, recon_loss, high_mask, low_mask, _high_emb, _low_emb)
                torch.cuda.empty_cache()
                gc.collect()
            del(dataloader)
            gc.collect()
        # validation , basically the same as training but
        '''
        torch.no_grad()
        no backward
        orthogonality loss here
        global decoder 
        sum the loss 
        '''
        val_loss = validate(model, graphs_list, val_idx)
        
        # compare ,and write a checkpoint to saved_models/
        if math.isfinite(val_loss) and val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs("saved_models", exist_ok=True)
            torch.save({
                'epoch': epoch,  # Save the current epoch number
                'model_state_dict': model.state_dict(),
                'decoder_state_dict': decoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': total_loss,  # Optionally, save the loss
                'args': args
            }, model_path)
            print(f"Model saved at Epoch: {epoch+1}")
        end_time = time.time()
        epoch_time_hours = (end_time - start_time) / 3600
        print(f"Epoch: {epoch + 1}, Loss: {total_loss}, Validation loss : {val_loss}, Best validation loss: {best_val_loss}, Time : {epoch_time_hours} hours")

        # Log to wandb
        if args.wandb and WANDB_AVAILABLE:
            log_dict = {
                "epoch": epoch + 1,
                "train/loss": total_loss,
                "val/loss": val_loss,
                "val/best_loss": best_val_loss,
                "time/epoch_hours": epoch_time_hours,
                "decoder/alpha_sigmoid": F.sigmoid(decoder.alpha).item(),
            }
            wandb.log(log_dict)

    # Fallback: save last epoch if no valid checkpoint was ever saved
    if best_val_loss >= 10e9:
        os.makedirs("saved_models", exist_ok=True)
        torch.save({
            'epoch': args.num_epochs - 1,
            'model_state_dict': model.state_dict(),
            'decoder_state_dict': decoder.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': total_loss,
            'args': args
        }, model_path)
        print('No valid validation loss; saved last epoch as fallback checkpoint.')

    if args.wandb and WANDB_AVAILABLE:
        wandb.finish()
'''
# interest:
how about 
    - model
    - dataloader
    - decoder
    - loss
    - optimizer
    - scheduler
    - metrics
    - callbacks
    - early stopping
    - logging
    - saving
    - loading
    - training
    - validation
'''
