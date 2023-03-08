import torch
from torch.utils.data import DataLoader
import numpy as np 
from tqdm import tqdm
from torch import optim
import matplotlib.pyplot as plt
from utils.options import handleOptions
from buildLikelihood import full_likelihood

def main():

    args, directory_name = handleOptions()
    # Now we decide how (if) we will use the gpu
    if args.device != 'cpu' and not torch.cuda.is_available():
        print("Warning, you tried to use cuda, but its not available. Will use the CPU")
        args.device = 'cpu'

    # If we use the cpu we dont use the whole UI (at psi)
    torch.set_num_threads(8)

    # now we get the data
    from utils.data import eftDataLoader
    signal_dataset = eftDataLoader( args )
    train,test  = torch.utils.data.random_split( signal_dataset, [0.7, 0.3], generator=torch.Generator().manual_seed(42))

    sm_weight,bsm_weight,features = test[:]
    likelihood = full_likelihood("/work/sesanche/EFT_mva_v2/ctg_regression.yaml")
    param_score = likelihood( features, {'ctg':2.}).detach().numpy()
    param_score = np.minimum(20,param_score)
    parametric_sm,bins,_  = plt.hist(param_score, weights=sm_weight.detach() , bins=200 , alpha=0.5, label="SM" , density=True)
    parametric_bsm,bins,_ = plt.hist(param_score, weights=bsm_weight.detach(), bins=bins, alpha=0.5, label="BSM", density=True)
    plt.legend()
    plt.savefig(f"{directory_name}/hist.png")
    plt.clf()

    dedicated = torch.load("20230308-141401_v1/network_bsm_weight_ctg_2_last.p", map_location=torch.device('cpu'))
    dedicated_score = dedicated(features).detach().numpy()
    dedicated_sm,bins,_  = plt.hist(dedicated_score, weights=sm_weight.detach() , bins=200 , alpha=0.5, label="SM" , density=True)
    dedicated_bsm,bins,_ = plt.hist(dedicated_score, weights=bsm_weight.detach(), bins=bins, alpha=0.5, label="BSM", density=True)
    plt.legend()
    plt.savefig(f"{directory_name}/hist_dedicated.png")
    plt.clf()

    
    cum_parametric_sm  = np.cumsum( parametric_sm  )
    cum_parametric_bsm = np.cumsum( parametric_bsm )

    cum_dedicated_sm  = np.cumsum( dedicated_sm  )
    cum_dedicated_bsm = np.cumsum( dedicated_bsm )

    plt.plot( cum_parametric_sm, cum_parametric_bsm, label="Parametric discriminator")
    plt.plot( cum_dedicated_sm, cum_dedicated_bsm, label="Dedicated discriminator")



if __name__=="__main__":
    main()
