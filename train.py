import torch
from torch.utils.data import DataLoader
import numpy as np 
from tqdm import tqdm
from torch import optim
import os 
import json
import matplotlib.pyplot as plt
import yaml
import time 

def main():
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--files",type=str, default="/pnfs/psi.ch/cms/trivcat/store/user/sesanche//EFT_mva/ttbar/NanoGen_tt_LO_SMEFTrwgt_v2_postprocess/*.root", help="List of files to process");
    parser.add_argument("--reload",  action='store_true', default=False, help="Force conversion of hdf to pytorch")
    parser.add_argument("--epochs",type=int, default=100, help="Number of epochs to train the net");
    parser.add_argument("--batch-size",type=int, default=64, help="Minibatch size");
    parser.add_argument("--term",  type=str, default=None, help="Train SM against a given term")
    parser.add_argument("--bsm-point",  type=str, default=None, help="Train SM against a given BSM point. Syntax is operator1=value2:operator2=value2:...")
    parser.add_argument("--device", type=str, default='cpu', help="Which device (cpu, gpu index) to use ");
    parser.add_argument("--name", type=str, default="v1", help="Name to store net.");
    parser.add_argument("--learning-rate", type=float, default=0.00000005, help="Optimizer learning rate")
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum for optimizer")
    parser.add_argument("--wc-list", type=str, default="sm,ctu1,cqd1,cqq13,ctu8,cqu1,cqq11,cqq83,ctd1,ctd8,ctg,ctq1,cqq81,cqu8,cqd8,ctq8", help="Comma-separated of WC in the sample (by order)")
    parser.add_argument("--features", type=str, default="Lep1_pt,Lep2_pt,Lep1_eta,Lep2_eta,Lep1_phi,Lep2_phi,nJet30,jet1_pt,jet2_pt", help="Comma-separated of WC in the sample (by order)")
    parser.add_argument("--forceRebuild", action="store_true", default=False, help="Force reproduction of torch tensors from rootfiles")
    parser.add_argument("--configuration-file", type=str, default=None, help="Load parameters from toml configuration file. The configuration file will be overriden by other command line options. The --name argument will always be taken from the command line option and the default")
    # to do parser.add_argument("--train-for-point", type=str, default=None, help="Train for a given EFT coeff value. If not specified, will get parametrized learning for the term specified in coef. Syntax operator_value")


    args = parser.parse_args()
    
    directory_name = time.strftime('%Y%m%d-%H%M%S') + "_" + args.name
    os.system(f"mkdir {directory_name}")


    # A bit of juggling with the configuration so we store it in yml files
    if args.configuration_file:
        with open(args.configuration_file) as f:
            config = yaml.safe_load(f.read())
    else:
        config = {}
    config = {**config, **vars(args)}
    
    with open(f"{directory_name}/config.yml","w") as f:
        f.write( yaml.dump(config)) 

    for op, val in config.items():
        setattr( args, op, val)

    # Now we decide how (if) we will use the gpu
    if args.device != 'cpu' and not torch.cuda.is_available():
        print("Warning, you tried to use cuda, but its not available. Will use the CPU")
        args.device = 'cpu'

    # If we use the cpu we dont use the whole UI (at psi)
    torch.set_num_threads(8)

    # all the stuff below should be configurable in the future
    # we get the model = net + cost function
    from models.net import Model
    model=Model(features = len(args.features.split(",")), device=args.device)

    # now we get the data
    from utils.data import eftDataLoader
    signal_dataset = eftDataLoader( args )
    train,test  = torch.utils.data.random_split( signal_dataset, [0.7, 0.3], generator=torch.Generator().manual_seed(42))
    dataloader     = DataLoader(  train  , batch_size=args.batch_size, shuffle=True)


    optimizer = optim.SGD(model.net.parameters(), lr=args.learning_rate, momentum=args.momentum)


    loss_train = []; loss_test=[]
    for epoch in range(args.epochs):
        
        for i,(sm_weight, bsm_weight, features) in tqdm( enumerate(dataloader)):
            optimizer.zero_grad()
            loss = model.cost_from_batch(features, sm_weight, bsm_weight, args.device)
            loss.backward()
            optimizer.step()
        loss_train.append( model.cost_from_batch(train[:][2] , train[:][0],  train[:][1], args.device).item())
        loss_test .append( model.cost_from_batch(test [:][2] , test [:][0],  test [:][1], args.device).item())

    plt.plot( range(args.epochs), loss_train, label="Training dataset")
    plt.plot( range(args.epochs), loss_test , label="Testing dataset")
    plt.savefig(f'{directory_name}/loss.png')
    plt.clf()

    

    
if __name__=="__main__":
    main()
