import torch
from torch.utils.data import DataLoader
import numpy as np 
from tqdm import tqdm
from torch import optim
import matplotlib.pyplot as plt
from utils.options import handleOptions
from utils.metrics import net_eval

def save_and_plot(net, loss_test, loss_train, label, directory_name, bsm_name, test):

    fig, ax = plt.subplots(1, 1, figsize=[8,8])
    torch.save(net, f'{directory_name}/network_{bsm_name}_{label}.p')

    ax.plot( range(len(loss_test)), loss_train, label="Training dataset")
    ax.plot( range(len(loss_test)), loss_test , label="Testing dataset")
    ax.legend()
    fig.savefig(f'{directory_name}/loss_{label}.png')
    plt.clf()

    fig, ax = plt.subplots(1, 1, figsize=[12,7])
    bins = np.linspace(0,1,100)
    sm_hist,_,_  = ax.hist(net(test[:][2]).flatten().detach().numpy(), 
                           weights=test[:][0], bins=bins, alpha=0.5, 
                           label='SM', density=True)
    
    bsm_hist,_,_ = ax.hist(net(test[:][2]).flatten().detach().numpy(), 
                           weights=test[:][1], bins=bins, alpha=0.5, 
                           label='BSM', density=True)
    ax.set_xlabel('Network Output', fontsize=12)
    ax.legend()
    fig.savefig(f'{directory_name}/net_out_{label}.png')
    plt.clf()
    
    roc, auc, a = net_eval(bins, sm_hist, bsm_hist, n_points=100)
    
    fig, ax = plt.subplots(1, 1, figsize=[8,8])
    ax.plot(roc[:,0], roc[:,1], label='Network Performance')
    ax.plot([0,1],[0,1], ':', label='Baseline')
    ax.legend()
    ax.set_title('Linear', fontsize=16)
    ax.set_xlabel('False Positive Rate', fontsize=14)
    ax.set_ylabel('True Positive Rate', fontsize=14)
    fig.savefig(f'{directory_name}/ROC_{label}.png')
    plt.clf()
    
    f = open(f'{directory_name}/performance_{label}.txt','w+')
    f.write(    
        'Area under ROC: ' + str(auc) + '\n' + 
        'Accuracy:       ' + str(a) + '\n'
    )
    f.close()

def main():

    args = handleOptions()


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
        
        for i,(sm_weight, bsm_weight, features) in tqdm(enumerate(dataloader)):
            optimizer.zero_grad()
            loss = model.cost_from_batch(features, sm_weight, bsm_weight, args.device)
            loss.backward()
            optimizer.step()
        loss_train.append( model.cost_from_batch(train[:][2] , train[:][0],  train[:][1], args.device).item())
        loss_test .append( model.cost_from_batch(test [:][2] , test [:][0],  test [:][1], args.device).item())
        if epoch%200==0: 
            save_and_plot( model.net, loss_test, loss_train, f"epoch_{epoch}", f"{args.name}", signal_dataset.bsm_name,
                         test)

    save_and_plot( model.net, loss_test, loss_train, "last", f"{args.name}", signal_dataset.bsm_name, test)
    
if __name__=="__main__":
    main()
