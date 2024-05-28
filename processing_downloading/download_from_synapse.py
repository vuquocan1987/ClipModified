# use synapse to download "syn10284975"

import synapseclient
import synapseutils
import os
import sys
import argparse

# token is in token_snapse.txt

argparser = argparse.ArgumentParser(description='Download data from synapse')
argparser.add_argument('--token-path', type=str, help='Synapse token')
argparser.add_argument('--synapse-id', type=str, help='Synapse ID')
argparser.add_argument('--output-dir', type=str, help='Output directory')

args = argparser.parse_args()
token_path = args.token_path
synapse_id = args.synapse_id
output_dir = args.output_dir

with open(token_path, "r") as f:
    token = f.read().strip()
print(token)
# def download_synapse_data(syn, syn_id, output_dir):

# download the folder "syn3376386" to data/clip/01_Multi_Atlas

syn = synapseclient.Synapse() 
syn.login(authToken=token) 
files = synapseutils.syncFromSynapse(syn, 'syn3376386',path = output_dir, followLink=True)


# python download_from_synapse.py --token-path ~/token_synapse.txt --synapse-id syn3376386 --output-dir /data/clip/01_Multi_Atlas
