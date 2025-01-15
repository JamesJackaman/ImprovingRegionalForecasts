import pickle
import subprocess
import time


MaxProcesses = 10
Processes = []

def checkrunning():
    for p in reversed(range(len(Processes))):
        if Processes[p].poll() is not None:
            del Processes[p]
    return len(Processes)


iterations = 1000 #number of data to generate

#generate
for i in range(iterations):
    print('i =', i, 'initialised')

    process = subprocess.Popen('python call_swe.py %s' % (i),
                               shell=True, stdout=subprocess.PIPE)
    Processes.append(process)

    while checkrunning()==MaxProcesses:
        time.sleep(1)

while checkrunning()!=0:
    time.sleep(1)
    
#load generated data
u_coarse = []
v_coarse = []
p_coarse = []
u_fine = []
v_fine = []
p_fine = []

for i in range(iterations):
    try:
        filename = 'tmp/swe%d.pickle' % i
        with open(filename,'rb') as file:
            outs = pickle.load(file)
            out_c = outs['coarse']
            out_f = outs['fine']

            u_coarse.append(out_c['u'])
            v_coarse.append(out_c['v'])
            p_coarse.append(out_c['p'])
            

            u_fine.append(out_f['u'])
            v_fine.append(out_f['v'])
            p_fine.append(out_f['p'])

            A = out_f['A']
            B = out_f['B']
            mass = out_f['mass']
            energy = out_f['energy']
            
    except Exception as e:
        print('Loading iterate %s failed with' % i)
        print('Error:' +str(e))
        
data = {'u_c': u_coarse,
        'v_c': v_coarse,
        'p_c': p_coarse,
        'u_f': u_fine,
        'v_f': v_fine,
        'p_f': p_fine,
        'A': A,
        'B': B,
        'mass': mass,
        'energy': energy}

#Save
filename = 'data_swe.pickle'
file = open(filename,'wb')
pickle.dump(data,file)
file.close()
