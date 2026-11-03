#global imports
import torch
import numpy as np
import copy
import os
import argparse

#local imports
import dataset

#abbrv some stuff
nn = torch.nn

#Choose device
if torch.cuda.is_available():
    device_name = 'cuda:0'
elif torch.backends.mps.is_available():
    device_name = 'mps'
else:
    device_name = 'cpu'
device = torch.device(device_name)

#Default parameters (LIKELY NOT BEING USED)
class default_parameters:
    def __init__(self):
        #Training parameters
        self.lr = 1e-3 #peak learning rate
        self.epochs=200
        self.batch_size = 16
        self.warmup = 5 #epochs of learning rate warm-up
        self.seed = 7
        
        #Network parameters
        self.variant = 'flow' #or 'noflow' (without the flow map) or 'unet' (UNet alone)
        self.layers = None #or the number of convolutions in a smaller UNet
        self.size1 = 2**3
        self.size2 = 2**4
        self.operator = 'fine' #fine flow map, or the controls 'identity' or 'random'

        #Data parameters
        self.stacksize = 10

        #Conservation?
        self.conserve = False
        self.sigma = 1 #Energy penalty

#Move data back to CPU
def detach(x):
    y = []
    for i in range(len(x)):
        y.append(x[i].cpu().detach().numpy())
    return y
        
#Apply a CNN to data scaled to be of order one
class scaled(nn.Module):
    def __init__(self,cnn,s_in,s_out):
        super().__init__()
        self.cnn = cnn
        self.s_in = s_in
        self.s_out = s_out

    def forward(self,x):
        return self.s_out * self.cnn(x / self.s_in)

#Define network
class network(nn.Module):
    def __init__(self,para):
        super().__init__()
        self.para = para
        #CNNs for interpolation
        ss = para.stacksize
        size1 = para.size1 * ss
        size2 = para.size2 * ss
        bias_switch = True
        if para.layers is None:
            self.layer1 = nn.Conv1d(ss,size1,kernel_size=3,padding=1,
                                    stride=1,bias=bias_switch,padding_mode='circular')
            self.layer2 = nn.Conv1d(size1,size2,kernel_size=5,padding=2,
                                    stride=1,bias=bias_switch,padding_mode='circular')
            self.layer3 = nn.Conv1d(size2,size2,kernel_size=7,padding=3,
                                    stride=1,bias=bias_switch,padding_mode='circular')
            self.layer4 = nn.Conv1d(size2,size1,kernel_size=5,padding=2,
                                    stride=1,bias=bias_switch,padding_mode='circular')
            self.layer5 = nn.Conv1d(size1,ss,kernel_size=3,padding=1,
                                    stride=1,bias=bias_switch,padding_mode='circular')

            self.cnn_u = nn.Sequential(
                self.layer1,
                nn.ReLU(),
                self.layer2,
                nn.ReLU(),
                self.layer3,
                nn.ReLU(),
                self.layer4,
                nn.ReLU(),
                self.layer5,
            )
        else:
            #Smaller UNet: L convolutions of kernel size 3 with size1 hidden channels
            channels = [ss] + [size1]*(para.layers-1) + [ss]
            layer_list = []
            for i in range(para.layers):
                if i > 0:
                    layer_list.append(nn.ReLU())
                conv = nn.Conv1d(channels[i],channels[i+1],kernel_size=3,padding=1,
                                 stride=1,bias=bias_switch,padding_mode='circular')
                layer_list.append(conv)
            self.cnn_u = nn.Sequential(*layer_list)
        self.cnn_v = copy.deepcopy(self.cnn_u)
        self.cnn_p = copy.deepcopy(self.cnn_u)

        #UNet alone has no UNet2
        if para.variant!='unet':
            self.cnn_u_ = copy.deepcopy(self.cnn_u)
            self.cnn_v_ = copy.deepcopy(self.cnn_u)
            self.cnn_p_ = copy.deepcopy(self.cnn_u)

        #Scale each component
        self.cnn_u = scaled(self.cnn_u,*para.scales['u'])
        self.cnn_v = scaled(self.cnn_v,*para.scales['v'])
        self.cnn_p = scaled(self.cnn_p,*para.scales['p'])
        if para.variant!='unet':
            self.cnn_u_ = scaled(self.cnn_u_,*para.scales['u'])
            self.cnn_v_ = scaled(self.cnn_v_,*para.scales['v'])
            self.cnn_p_ = scaled(self.cnn_p_,*para.scales['p'])

        
    def index(self,func,i):
        return torch.unsqueeze(func[:,i,:],1)
        
    def forward(self,x):
        #Unpack input
        u, v, p = x        

        #Interpolate
        u = self.cnn_u(u) + u
        v = self.cnn_v(v) + v
        p = self.cnn_p(p) + p

        #UNet alone
        if self.para.variant=='unet':
            return u, v, p

        #Dynamic step
        udim = u.shape[2]
        vdim = v.shape[2]
        pdim = p.shape[2]
        z = torch.cat((u,v,p),2)

        if self.para.variant=='flow':
            for i in range(self.para.stacksize-1):
                rhs = torch.einsum('ij,bj -> bi', self.para.Ainv @ self.para.B, z[:,i,:])
                rhs = rhs.unsqueeze(1)
                z[:,i+1,:] = rhs.squeeze(1)
            
        #Resplit data
        u = u + self.cnn_u_(z[:,:,:udim])
        v = v + self.cnn_v_(z[:,:,udim:udim+vdim])
        p = p + self.cnn_p_(z[:,:,udim+vdim:])

        return u, v, p


#Define training routine
def train(model,trainloader,sp):
    def mseLoss(output, target):
        o_u, o_v, o_p = output
        t_u, t_v, t_p = target
        out = torch.mean((o_u-t_u)**2) + torch.mean((o_v-t_v)**2) + torch.mean((o_p-t_p)**2)
        return out/3

    def feLoss(output, target):
        #Stack functions
        output = torch.cat((output[0],output[1],output[2]),2)
        target = torch.cat((target[0],target[1],target[2]),2)
        diff = output - target
        L2 = torch.einsum('bsn, nm, bsm -> bs', diff, sp.mass, diff)
        
        return torch.mean(L2)

    def constrainedLoss(output, target, initial_masses, initial_energies):
        #Stack functions
        output = torch.cat((output[0],output[1],output[2]),2)
        target = torch.cat((target[0],target[1],target[2]),2)
        diff = output - target
        L2 = torch.einsum('bsn, nm, bsm -> bs', diff, sp.mass, diff)
        reg_mass = torch.abs(torch.einsum('n,nm, bsm -> bs',
                                 torch.ones(sp.mass.shape[0]).to(device),
                                 sp.mass, output)
                    - initial_masses.unsqueeze(1))
        reg_energy = torch.abs(torch.einsum('bsn, nm, bsm -> bs',
                                   output, sp.energy, output)
                      - initial_energies.unsqueeze(1))

        error = L2 + sp.sigma * reg_energy
        
        return torch.mean(error)
    
    # Choose loss critera
    criterion = feLoss
    # Overwrite if we care about conservation
    if sp.conserve:
        criterion = constrainedLoss

    # Choose optimiser
    optimiser = torch.optim.Adam(model.parameters(),lr=sp.lr,weight_decay=0)
    # Choose learning rate scheduler (linear warm-up, then cosine decay)
    steps = sp.epochs * len(trainloader)
    warmup = sp.warmup * len(trainloader)
    def schedule(step):
        if step < warmup:
            return (step+1) / warmup
        return 0.5 * (1 + np.cos(np.pi * (step-warmup) / (steps-warmup)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimiser,schedule)
    
    epoch = 0
        
    #Loop over epochs
    count = 0
    while epoch < sp.epochs:
        epoch+=1
        running_loss = 0
        for i, data in enumerate(trainloader):
            
            inputs, labels = data #If slow these are really on GPU

            #Get mass and energy
            if sp.conserve==True:
                z = torch.cat((inputs),2)
                initial_masses = torch.einsum('n,nm, bsm -> bs',
                                      torch.ones(sp.mass.shape[0]).to(device),
                                      sp.mass, z)[:,0]
                initial_energies = torch.einsum('bsn, nm, bsm -> bs',
                                                z, sp.energy, z)[:,0]

            optimiser.zero_grad() #Initialise gradients as None
            
            outputs = model(inputs) #Run model

            #Add initial masses and energies if we care about conservation
            if sp.conserve:
                loss = criterion(outputs,labels,initial_masses,initial_energies)
            else:
                loss = criterion(outputs,labels) #Define loss
        
            (loss / sp.ref).backward() #Backpropagation
            optimiser.step() #Optimise
            scheduler.step() #Learning rate scheduler

            running_loss += loss.item()

            count += len(inputs)
            
            if i%20 == 0 and i > 0:
                print(f'Loss [{epoch}, {i}](epoch, minibatch): ', running_loss / 20)
                running_loss = 0.0

    print('Training finished')

    torch.save(model.state_dict(),sp.name+'.pt')

    print('Model saved')
    
    return model


def model_run(sp):
    #Name the run (as in plot.py)
    sigma = 0
    if sp.conserve:
        sigma = sp.sigma
    sp.name = 'results/%s_sigma%g_seed%d' % (sp.variant,sigma,sp.seed)
    if sp.layers is not None:
        sp.name += '_L%d' % sp.layers
    if (sp.size1,sp.size2)!=(2**3,2**6):
        sp.name += '_size%d-%d' % (sp.size1,sp.size2)
    if sp.operator!='fine':
        sp.name += '_' + sp.operator
    #Skip finished runs
    if os.path.isfile(sp.name+'.npz'):
        print(sp.name, 'has already finished')
        return
    os.makedirs('results',exist_ok=True)

    #Fix seed
    torch.manual_seed(sp.seed)

    #Split data into test and training
    trainset, testset = dataset.all_data(device=device,stacksize=sp.stacksize)
    #Get the mass matrix and put on GPU (for custom loss function)
    sp.mass = torch.from_numpy(trainset.invariant['mass'].todense()).float().to(device)
    sp.energy = torch.from_numpy(trainset.invariant['energy'].todense()).float().to(device)

    #Copy A and B matrices to dictionary from dataset
    sp.Ainv = trainset.Ainv
    sp.B = trainset.B

    #Standard deviation of the coarse data, and of its difference from the fine data
    sp.scales = {}
    sp.scales['u'] = (trainset.u_in.std().item(),(trainset.u_out-trainset.u_in).std().item())
    sp.scales['v'] = (trainset.v_in.std().item(),(trainset.v_out-trainset.v_in).std().item())
    sp.scales['p'] = (trainset.p_in.std().item(),(trainset.p_out-trainset.p_in).std().item())

    #Controls: the identity or S Q S^-1 (Q random orthogonal, S the scales) as the flow map
    if sp.operator=='identity':
        sp.Ainv = torch.eye(len(sp.B)).to(device)
        sp.B = torch.eye(len(sp.B)).to(device)
    elif sp.operator=='random':
        s = np.concatenate((np.full(trainset.u_in.shape[2],sp.scales['u'][0]),
                            np.full(trainset.v_in.shape[2],sp.scales['v'][0]),
                            np.full(trainset.p_in.shape[2],sp.scales['p'][0])))
        Q, R = np.linalg.qr(np.random.default_rng(0).standard_normal((len(s),len(s))))
        Q = Q * np.sign(np.diag(R)) #uniformly distributed over the orthogonal matrices
        sp.Ainv = torch.eye(len(s)).to(device)
        sp.B = torch.from_numpy(s[:,None] * Q / s[None,:]).float().to(device)

    #Mass and energy in double precision (off device)
    mass = trainset.invariant['mass'].toarray()
    energy = trainset.invariant['energy'].toarray()

    #Normalise the loss by the average squared L2 error of the coarse training data
    coarse = torch.cat((trainset.u_in,trainset.v_in,trainset.p_in),2).cpu().double().numpy()
    fine = torch.cat((trainset.u_out,trainset.v_out,trainset.p_out),2).cpu().double().numpy()
    L2 = np.einsum('bsn, nm, bsm -> bs', coarse-fine, mass, coarse-fine, optimize=True)
    sp.ref = float(np.mean(L2))

    #Initialise network
    model = network(sp)
    model.to(device)
    params = sum(q.numel() for q in model.parameters())
    print('Parameters =', params)

    #Batch data and shuffle the training data
    trainloader = torch.utils.data.DataLoader(trainset,batch_size=sp.batch_size,
                                              shuffle=True,num_workers=0)
    testloader = torch.utils.data.DataLoader(testset,batch_size=sp.batch_size,
                                             shuffle=False,num_workers=0)

    #Train network
    train(model,trainloader,sp)

    #Run model on all validation data
    X, Y, Z = [], [], []
    with torch.no_grad():
        for inputs, labels in testloader:
            X.append(torch.cat(inputs,2).cpu().double().numpy()) #input
            Y.append(torch.cat(labels,2).cpu().double().numpy()) #truth
            Z.append(torch.cat(model(inputs),2).cpu().double().numpy()) #output
    X, Y, Z = np.concatenate(X), np.concatenate(Y), np.concatenate(Z)

    #Compute squared L2 errors and norm for each run and time level
    L2err = np.einsum('bsn, nm, bsm -> bs', Z-Y, mass, Z-Y, optimize=True)
    L2coarse = np.einsum('bsn, nm, bsm -> bs', X-Y, mass, X-Y, optimize=True)
    L2norm = np.einsum('bsn, nm, bsm -> bs', Y, mass, Y, optimize=True)

    #Compute relative deviation from the initial energy
    Energy = np.einsum('bsn, nm, bsm -> bs', Z, energy, Z, optimize=True)
    Energy0 = np.einsum('bn, nm, bm -> b', X[:,0], energy, X[:,0], optimize=True)
    dEnergy = np.abs(Energy - Energy0[:,None]) / Energy0[:,None]

    print('L2err =', np.mean(L2err))
    print('L2coarse =', np.mean(L2coarse))
    print('dEnergy =', np.mean(dEnergy))

    #Save results (for plot.py)
    np.savez(sp.name+'.npz',L2err=L2err,L2coarse=L2coarse,L2norm=L2norm,dEnergy=dEnergy,
             params=params)


if __name__=="__main__":

    # model_run(default_parameters())

    # input('test run finished')
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', type=int, nargs='+', default=[1,2,7],
                        help = 'Seeds to run (to split the runs between processes)')
    args = parser.parse_args()

    #Experiments as (variant, sigma, layers, size1, operator)
    experiments = []
    #Table 1
    for sigma in [0,0.1,1,10]:
        experiments.append(('flow',sigma,None,2**3,'fine'))
    #Figure 3
    for layers in [2,3,4,5,None]:
        for variant in ['flow','noflow','unet']:
            experiments.append((variant,0,layers,2**3,'fine'))
    #Appendix B: wider networks
    for layers in [2,3]:
        for variant in ['flow','noflow']:
            experiments.append((variant,0,layers,2**5,'fine'))
    #Appendix B: UNet alone with about as many parameters as the network with the flow map
    experiments.append(('unet',0,2,2**4,'fine'))
    experiments.append(('unet',0,3,12,'fine'))
    #Appendix B: the flow map replaced by the identity or a random map
    for layers in [2,3]:
        for operator in ['identity','random']:
            experiments.append(('flow',0,layers,2**3,operator))

    for seed in args.seeds:
        for variant, sigma, layers, size1, operator in experiments:

            print('Now running %s with sigma=%s, layers=%s, size1=%s, operator=%s, seed=%s'
                  % (variant,sigma,layers,size1,operator,seed))
                    
            #Initialise parameters
            class super_parameters:
                def __init__(self):

                    #Training parameters
                    self.lr = 1e-3 #peak learning rate
                    self.epochs=300
                    self.batch_size = 16
                    self.warmup = 5 #epochs of learning rate warm-up
                    self.noise = 1e-12
                    self.seed = seed

                    #Network parameters
                    self.variant = variant
                    self.layers = layers
                    self.size1 = size1
                    self.size2 = 2**6
                    self.operator = operator

                    #Data parameters
                    self.stacksize = 10

                    #Conserve stuff?
                    self.conserve = sigma > 0
                    self.sigma = sigma #Level of energy regularisation

            #Run model
            sp = super_parameters()
            model_run(sp)
