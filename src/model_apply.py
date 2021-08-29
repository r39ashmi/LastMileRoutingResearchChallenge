## Author: RASHMI
from os import path
import json
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

from utils.functions import torch_load_cpu,move_to
from utils.problem_amz_routing import CVRP
from utils.models.attention_model import AttentionModel,set_decode_type

torch.manual_seed(400)
BASE_DIR = path.dirname(path.dirname(path.abspath(__file__)))
distribution='val'
batch_size=1
device=torch.device("cpu")
epoch_start=0
batch_size=1
embedding_dim=128
hidden_dim=128
n_encode_layers=3
normalization='batch'
tanh_clipping=10
checkpoint_encoder=True
lr_model=1e-4
lr_decay=1.0
load_path=path.join(BASE_DIR, 'data/model_build_outputs/final_mdl_6.pt')
checkpoint_epochs=5
save_dir=path.join(BASE_DIR, 'data/model_build_outputs/')

# Read input data
print('Reading Input Data')

filepath=path.join(BASE_DIR, 'data/model_apply_inputs/new_route_data.json')
pack_data_path=path.join(BASE_DIR, 'data/model_apply_inputs/new_package_data.json')
travel_times_data_path=path.join(BASE_DIR, 'data/model_apply_inputs/new_travel_times.json')
output_filename=path.join(BASE_DIR, 'data/model_apply_outputs/lat_lng_info_val.pkl')
output_filename2=path.join(BASE_DIR,'data/model_build_outputs/classes.npy')


problem=CVRP()
model=AttentionModel(embedding_dim,hidden_dim,problem,n_encode_layers=n_encode_layers,mask_inner=True,mask_logits=True,normalization=normalization,tanh_clipping=tanh_clipping,checkpoint_encoder=checkpoint_encoder).to(device)	


### Load model
if load_path is not None:
	print('  [*] Loading data from {}'.format(load_path))
	load_data = torch_load_cpu(load_path)
	model.load_state_dict(load_data['model'])

##Format data
val_dataset = problem.make_dataset(distribution='val')		
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, num_workers=1)

# Put model in train mode!
model.eval()
set_decode_type(model, "greedy")

def sort_by_key(stops, sort_by):
    # Serialize keys as id into each dictionary value and make the dict a list
    stop_l=[value for value in stops]

    # Sort the stops list by the key specified when calling the sort_by_key func
    ordered_stop_list=dict(sorted(stop_l[sort_by].items(), key=lambda item: item[sort_by])) #sorted(stops_list, key=lambda x: x[sort_by])

    # Serialize back to dictionary format with output order as the values
    return ordered_stop_list #{i:ordered_stop_list_ids.index(i) for i in ordered_stop_list_ids}


def propose_all_routes(prediction_routes, sort_by):    
    return {key:{'proposed':sort_by_key(stops=value.values(), sort_by=sort_by)} for key, value in prediction_routes.items()}
    
final_seq={ }
for batch_id, batch in enumerate(tqdm(val_dataloader, disable=False)):
	stop_seq=batch['stop_seq']
	rt_id=batch['rt_id']
	depot=batch['st_code']
	del batch['stop_seq']
	del batch['rt_id']
	del batch['st_code']
	x = move_to(batch, device)

	# Evaluate model, get costs and log probabilities
	pi = model(x,distribution='val')
	final_seq[rt_id[0]]={}
	final_seq[rt_id[0]]['proposed']={}
	final_seq[rt_id[0]]['proposed'][depot[0]]=0
	for st_ind in range(len(pi.squeeze())):
	    ind=stop_seq[pi.squeeze()[st_ind]-1][0]
	    final_seq[rt_id[0]]['proposed'][ind]=st_ind+1 
sort_by=0
output=propose_all_routes(final_seq, sort_by)


# Write output data
output_path=path.join(BASE_DIR, 'data/model_apply_outputs/proposed_sequences.json')
with open(output_path, 'w') as out_file:
    json.dump(output, out_file)
    print("Success: The '{}' file has been saved".format(output_path))

print('Done!')
