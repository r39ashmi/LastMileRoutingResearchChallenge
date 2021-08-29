## Author: RASHMI
import os
from os import path
import time
from tqdm import tqdm
import torch
import torch.optim as optim
from tensorboard_logger import Logger as TbLogger
from torch.utils.data import DataLoader

from utils.functions import torch_load_cpu
from utils.problem_amz_routing import CVRP
from utils.models.attention_model import AttentionModel, set_decode_type
from utils.train import train_batch, get_inner_model

st_time=time.time()
torch.manual_seed(400)

# Get Directory
BASE_DIR = path.dirname(path.dirname(path.abspath(__file__)))
print(BASE_DIR)
# Read input data
print('Reading Input Data')
filepath = 'data/model_build_inputs/route_data.json'
pack_data = 'data/model_build_inputs/package_data.json'
actual_data_path = 'data/model_build_inputs/actual_sequences.json'
travel_data_path = 'data/model_build_inputs/travel_times.json'
output_filename = 'data/model_build_outputs/lat_lng_info.pkl'
filepath = path.join(BASE_DIR, filepath)
pack_data_path = path.join(BASE_DIR, pack_data)
actual_data_path = path.join(BASE_DIR, actual_data_path)
travel_data_path = path.join(BASE_DIR, travel_data_path)
output_filename = path.join(BASE_DIR, output_filename)
output_filename2=path.join(BASE_DIR,'data/model_build_outputs/classes.npy')
	
print("READING Done")
# Hyper-parameters
device=torch.device("cpu")
epoch_start=0
n_epochs=30
batch_size=1
embedding_dim=128
hidden_dim=128
n_encode_layers=3
normalization='batch'
tanh_clipping=10
checkpoint_encoder=True
lr_model=1e-4
lr_decay=1.0
resume=None
load_path=None

checkpoint_epochs=1
save_dir=path.join(BASE_DIR, 'data/model_build_outputs/')
#Formatting data for the problem
problem=CVRP()
st=time.time()
training_dataset = problem.make_dataset(distribution='train')
epoch_size=len(training_dataset)

training_dataloader = DataLoader(training_dataset, batch_size=batch_size,shuffle = True) 

# Attention model initialization
model = AttentionModel(embedding_dim,hidden_dim,problem,n_encode_layers=n_encode_layers,mask_inner=True,mask_logits=True,normalization=normalization,     tanh_clipping=tanh_clipping,checkpoint_encoder=checkpoint_encoder).to(device)


load_data = {}
assert load_path is None or resume is None, "Only one of load path and resume can be given"
load_path = load_path if load_path is not None else resume
if load_path is not None:
	print('  [*] Loading data from {}'.format(load_path))
	load_data = torch_load_cpu(load_path)
model_ = get_inner_model(model)
model_.load_state_dict({**model_.state_dict(), **load_data.get('model', {})})


# Initialize optimizer
optimizer = optim.Adam([{'params': model.parameters(), 'lr': lr_model}])

# Load optimizer state
if 'optimizer' in load_data:
	optimizer.load_state_dict(load_data['optimizer'])
	for state in optimizer.state.values():
		for k, v in state.items():
			# if isinstance(v, torch.Tensor):
			if torch.is_tensor(v):
			    state[k] = v.to(device)

# Initialize learning rate scheduler, decay by lr_decay once per epoch!
lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: lr_decay ** epoch)


total_time=time.time() - st_time
for epoch in range(epoch_start, epoch_start + n_epochs):
    model.train()
    set_decode_type(model, "sampling")
    step=epoch * (epoch_size // batch_size)
    start_time = time.time()
    overall_cost = 0
    overall_CE = 0
    for batch_id, batch in enumerate(tqdm(training_dataloader, disable=False)):
        start_time1 = time.time()    
        CE=train_batch(
            model,
            optimizer,
            epoch,
            batch_id,
            step,
            batch,
            TbLogger,
            device,1.0
        )
        overall_CE+=CE
        step += 1
    
    
    epoch_duration = time.time() - start_time   
    total_time+=epoch_duration
    print("Finished epoch {}, took {} s".format(epoch, time.strftime('%H:%M:%S', time.gmtime(epoch_duration))))
    print("Over all  CE {}".format(CE))
    print('Saving model and state...')
    torch.save(
            {
                'model': get_inner_model(model).state_dict(),
                'optimizer': optimizer.state_dict(),
                'rng_state': torch.get_rng_state(),
                'cuda_rng_state': torch.cuda.get_rng_state_all(),
            },
            os.path.join(save_dir, 'final_mdl_{}.pt'.format(epoch))
        )
    if total_time+epoch_duration > 43203 or epoch==n_epochs-1:
        break

    lr_scheduler.step()
