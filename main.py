#global imports
import torch
import numpy as np
import copy

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

#Fix seed
torch.manual_seed(7)

#Default parameters (LIKELY NOT BEING USED)
class default_parameters:
    def __init__(self):
        #Training parameters
        self.lr = 1e-3
        self.epochs=200
        self.batch_size = 16
        self.step_size = 30
        self.gamma = 0.1
        self.tol = 1e-6 #convergence tol
        
        #Network parameters
        self.size1 = 2**3
        self.size2 = 2**4

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
        self.cnn_v = copy.deepcopy(self.cnn_u)
        self.cnn_p = copy.deepcopy(self.cnn_u)

        self.cnn_u_ = copy.deepcopy(self.cnn_u)
        self.cnn_v_ = copy.deepcopy(self.cnn_u)
        self.cnn_p_ = copy.deepcopy(self.cnn_u)

        
    def index(self,func,i):
        return torch.unsqueeze(func[:,i,:],1)
        
    def forward(self,x):
        #Unpack input
        u, v, p = x        

        #Interpolate
        u = self.cnn_u(u)
        v = self.cnn_v(v)
        p = self.cnn_p(p)

        #Dynamic step
        udim = u.shape[2]
        vdim = v.shape[2]
        pdim = p.shape[2]
        z = torch.cat((u,v,p),2)

        for i in range(self.para.stacksize-1):
            rhs = torch.einsum('ij,bj -> bj', self.para.Ainv @ self.para.B, z[:,i,:])
            rhs = rhs.unsqueeze(1)
            z[:,i+1,:] = rhs.squeeze(1)
            
        #Resplit data
        u = u + self.cnn_u_(z[:,:,:udim])
        v = v + self.cnn_v_(z[:,:,udim+1:udim+vdim+1])
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
    scheduler = torch.optim.lr_scheduler.StepLR(optimiser,
                                                step_size=sp.step_size,gamma=sp.gamma)
    
    lossVal = 1
    epoch = 0
        
    #Loop over epochs
    count = 0
    while epoch < sp.epochs and lossVal > sp.tol:
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
        
            loss.backward() #Backpropagation
            optimiser.step() #Optimise

            running_loss += loss.item()

            count += len(inputs)
            
            if i%20 == 0 and i > 0:
                print(f'Loss [{epoch}, {i}](epoch, minibatch): ', running_loss / 20)
                running_loss = 0.0

        lossVal = loss.item()
        scheduler.step() #Learning rate scheduler

    print('Training finished')

    torch.save(model.state_dict(),"learn_flow.pt")

    print('Model saved')
    
    return model


def model_run(sp):
    #Split data into test and training
    trainset, testset = dataset.all_data(device=device,stacksize=sp.stacksize)
    #Get the mass matrix and put on GPU (for custom loss function)
    sp.mass = torch.from_numpy(trainset.invariant['mass'].todense()).float().to(device)
    sp.energy = torch.from_numpy(trainset.invariant['energy'].todense()).float().to(device)

    #Copy A and B matrices to dictionary from dataset
    sp.Ainv = trainset.Ainv
    sp.B = trainset.B

    #Initialise network
    model = network(sp)
    model.to(device)

    #Batch and shuffle data
    trainloader = torch.utils.data.DataLoader(trainset,batch_size=sp.batch_size,
                                              shuffle=True,num_workers=0)
    testloader = torch.utils.data.DataLoader(testset,batch_size=sp.batch_size,
                                             shuffle=True,num_workers=0)

    #Train network
    train(model,trainloader,sp)

    #Load model
    model.load_state_dict(torch.load("learn_flow.pt"))

    #Move mass and energy off device
    mass = np.array(detach(sp.mass))
    energy = np.array(detach(sp.energy))

    for i in range(10):
        X, Y = next(iter(testloader))

        Z = model(X)

        U_, V_, P_ = detach(X) #input
        U_true, V_true, P_true = detach(Y) #truth
        U, V, P = detach(Z) #output

        U_err, V_err, P_err = (U-U_true), (V-V_true), (P-P_true)

        #Combine
        z = np.concatenate((U_err, V_err, P_err),2)
        
        #Compute L2 error
        L2err = np.mean(np.einsum('bsn, nm, bsm -> bs', z, mass, z),axis=0)
        print('L2err =', L2err)
        
        #Compute MSE errors
        mse = (np.mean(U_err**2, axis=(0,2))
               + np.mean(V_err**2, axis=(0,2))
               + np.mean(P_err**2, axis=(0,2)))/3
        print('MSE =', mse)

        #Compute deviation in invariants
        Mass = np.mean(np.einsum('n, nm, bsm -> bs', np.ones_like(z[0,0,:]), mass, z),
                       axis=0)
        Energy = np.mean(np.einsum('bsn, nm, bsm -> bs', z, energy, z),
                         axis=0)

        print('Mass', Mass)
        print('Energy', Energy)

        dMass = np.abs(np.ones_like(Mass)*Mass[0] - Mass[:])
        dEnergy = np.abs(np.ones_like(Energy)*Energy[0] - Energy[:])    


if __name__=="__main__":

    # model_run(default_parameters())

    # input('test run finished')
    
    for conserve in [False,True]:
        if conserve==True:
            sigmas = [0.1,1,10]
        else:
            sigmas = [-1]
        for sigma in sigmas:
            for size1 in [2**3]:
                for size2 in [2**6]:

                    print('Now running conserve=%s with sigma=%s' % (conserve,sigma))
                    
                    #Initialise parameters
                    class super_parameters:
                        def __init__(self):

                            #Training parameters
                            self.lr = 1e-3
                            self.epochs=300
                            self.batch_size = 16
                            self.step_size = 30
                            self.gamma = 0.1
                            self.tol = 1e-6 #convergence tol
                            self.noise = 1e-12

                            #Network parameters
                            self.size1 = size1
                            self.size2 = size2

                            #Data parameters
                            self.stacksize = 10

                            #Conserve stuff?
                            self.conserve = conserve
                            self.sigma = sigma #Level of energy regularisation

                    #Run model
                    sp = super_parameters()
                    model_run(sp)
