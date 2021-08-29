from torch.utils.data import Dataset
import torch
import os
import numpy as np
import json
import sys
from .state_lmrc import StateLMRC
torch.manual_seed(400)

def normalize_matrix(mat,mat1):
    '''
    Normalizes cost matrix.

    Parameters
    ----------
    mat : dict
        Cost matrix.

    Returns
    -------
    new_mat : dict
        Normalized cost matrix.

    '''
    avg_time=torch.mean(torch.cat((mat.squeeze(),mat1),axis=-1))
    std_time=torch.std(torch.cat((mat.squeeze(),mat1),axis=-1))


    new_mat=(mat-avg_time)/std_time
    min_new_time=torch.min(new_mat)
    new_mat1=new_mat-min_new_time


    new_mat2=(mat1-avg_time)/std_time
    min_new_time=torch.min(new_mat2)
    new_mat2=new_mat2-min_new_time

    return new_mat1,new_mat2

def total_travel_time(time_data,path):
    if len(path) > 1:
        ind1,ind2=path[0],path[1]
        dist=total_travel_time(time_data,path[1:])
    else:
        return 0 
    return dist+time_data[:,ind1-1,ind2-1]

def get_short_distance(adj_mat,depot_ad):
        
        beam=10 # BEAM        
        visited=torch.zeros(beam,adj_mat.shape[1], dtype=torch.bool)
        #SOURCE to DESTINATION
        visited[torch.arange(beam),torch.argsort(depot_ad[:,0])[:beam]]=True
        prev_a=torch.cat((torch.arange(beam)[:,None],torch.argsort(depot_ad[:,0])[:beam][:,None]),axis=1)
        dist=torch.sort(depot_ad[:,0]).values[:beam].squeeze()
        
        for i in range(adj_mat.shape[1]-1):
            all_dist=torch.zeros((beam*beam))
            all_nodes=torch.zeros((beam*beam),dtype=int)
            for beam_ind in range(beam):
                all_nodes[beam_ind*beam:(beam_ind+1)*beam]=torch.sort(adj_mat[:,prev_a[beam_ind,1],visited[beam_ind]==False]).indices[:,:beam]
                all_dist[beam_ind*beam:(beam_ind+1)*beam]=dist[beam_ind]+torch.sort(adj_mat[:,prev_a[beam_ind,1],visited[beam_ind]==False]).values[:,:beam]
            dist=torch.sort(all_dist).values[:beam]
            
            visited=visited[torch.sort(all_dist).indices[:beam]%(beam-1),:]
            visited[torch.arange(beam),all_nodes[torch.sort(all_dist).indices[:beam]]]=True
            prev_a=torch.cat((torch.arange(beam)[:,None],all_nodes[torch.sort(all_dist).indices[:beam]][:,None]),axis=1)
        dist=dist+depot_ad[prev_a[:,1],1]
        return torch.min(dist)               
        #dist+=torch.sort(depot_ad[:,:,1]).values[:,:beam-1]

def score(actual,sub,cost_mat,g=1000):
    return erp_per_edit(actual,sub,cost_mat,g)

def erp_per_edit(actual,sub,matrix,g=1000):
    total=erp_per_edit_helper(actual,sub,matrix,g,None)
    return total

def gap_sum(path,g):
    res=0
    for p in path:
        res+=g
    return res


def dist_erp(p_1,p_2,mat,g=1000):

    if p_1=='gap' or p_2=='gap':
        dist=g
    else:
        dist=mat[int(p_1),int(p_2)]
    return dist

def erp_per_edit_helper(actual,sub,matrix,g=1000,memo=None):
    if memo==None:
        memo={}
    dist=0
    for head_actual,head_sub in zip(actual,sub):
        option_1=dist_erp(head_actual,head_sub,matrix,g)
        option_2=dist_erp(head_actual,'gap',matrix,g)
        option_3=dist_erp(head_sub,'gap',matrix,g)
        dist+=min(option_1,option_2,option_3)
        
    #memo[(actual_tuple,sub_tuple)]=(d,count)
    return dist

def seq_dev(actual,sub):
    actual=actual[1:]
    sub=sub[1:]
    #print(actual)
    #print(sub)
    comp_list=[]
    for i in sub:
        #print(i)
        #print((actual == i).nonzero(as_tuple=True)[0][0])
        comp_list.append((actual == i).nonzero(as_tuple=True)[0][0])
    comp_sum=0
    for ind in range(1,len(comp_list)):
        comp_sum+=abs(comp_list[ind]-comp_list[ind-1])-1
    n=len(actual)
    return (2/(n*(n-1)))*comp_sum

class CVRP(object):

    NAME = 'cvrp'  # Capacitated Vehicle Routing Problem

    VEHICLE_CAPACITY = 1.0  # (w.l.o.g. vehicle capacity is 1, demands should be scaled)
    
    
    @staticmethod
    def get_costs_lat(dataset, pi):
        batch_size, graph_size = dataset['demand'].size()
        # Check that tours are valid, i.e. contain 0 to n -1
        sorted_pi = pi.data.sort(1)[0]

        # Sorting it should give all zeros at front and then 1...n
        assert (
            torch.arange(1, graph_size + 1, out=pi.data.new()).view(1, -1).expand(batch_size, graph_size) ==
            sorted_pi[:, -graph_size:]
        ).all() and (sorted_pi[:, :-graph_size] == 0).all(), "Invalid tour"

        # Visiting depot resets capacity so we add demand = -capacity (we make sure it does not become negative)
        demand_with_depot = torch.cat(
            (
                torch.full_like(dataset['demand'][:, :1], -CVRP.VEHICLE_CAPACITY),
                dataset['demand']
            ),
            1
        )
        d = demand_with_depot.gather(1, pi)

        #used_cap = torch.zeros_like(dataset['demand'][:, 0])
        '''
        for i in range(pi.size(1)):
            used_cap += d[:, i]  # This will reset/make capacity negative if i == 0, e.g. depot visited
            # Cannot use less than 0
            used_cap[used_cap < 0] = 0
            assert (used_cap <= CVRP.VEHICLE_CAPACITY + 1e-5).all(), "Used more than capacity"
        '''
        # Gather dataset in order of tour
        loc_with_depot = torch.cat((dataset['depot'][:, None, :], dataset['loc']), 1)
        d = loc_with_depot.gather(1, pi[..., None].expand(*pi.size(), loc_with_depot.size(-1)))

        # Length is distance (L2-norm of difference) of each next location to its prev and of first and last to depot
        return (
            (d[:, 1:] - d[:, :-1]).norm(p=2, dim=2).sum(1)
            + (d[:, 0] - dataset['depot']).norm(p=2, dim=1)  # Depot to first
            + (d[:, -1] - dataset['depot']).norm(p=2, dim=1)  # Last to depot, will be 0 if depot is last
        ), None

            
    @staticmethod
    def get_costs(dataset, pi,actual_path):
        #batch_size, graph_size = dataset['demand'].size()
        travel_times,st_times=normalize_matrix(dataset['travel_times'],dataset['station_times'].squeeze())
        st_data1=st_times[:,0]        
        st_data2=st_times[:,1]
        adj=torch.cat((st_data1[None,:],travel_times.squeeze()),0)
        adj=torch.cat((torch.cat((torch.tensor([0])[:,None],st_data2[:,None]),0),adj),1)
        #distance=total_travel_time(travel_times,pi[0])+st_times[pi[0][0]-1,0]+st_times[pi[0][-1]-1,1]
        
        #actual_path=torch.tensor(actual_path,dtype=int)
        #print(np.shape(actual_path))
        #act_distance=total_travel_time(travel_times,actual_path)+st_times[actual_path[0]-1,0]+st_times[actual_path[-1]-1,1]
        return score(actual_path,torch.cat((torch.Tensor([0]),pi.squeeze())),adj),seq_dev(actual_path,torch.cat((torch.Tensor([0]),pi.squeeze()))),None 

    @staticmethod
    def make_dataset(*args, **kwargs):
        return VRPDataset(*args, **kwargs)

    @staticmethod
    def make_state(*args,distribution='val', **kwargs):
        return StateLMRC.initialize(*args, distribution=distribution,**kwargs)
    

def read_json_data(filepath):
    '''
    Loads JSON file and generates a dictionary from it.

    Parameters
    ----------
    filepath : str
        Path of desired file.

    Raises
    ------
    JSONDecodeError
        The file exists and is readable, but it does not have the proper
        formatting for its place in the inputs of evaluate.

    Returns
    -------
    file : dict
        Dictionary form of the JSON file to which filepath points.

    '''
    try:
        with open(filepath, newline = '') as in_file:
            file=json.load(in_file)
            in_file.close()
    except FileNotFoundError:
        print("The '{}' file is missing!".format(filepath))
        sys.exit()
    except Exception as e:
        print("Error when reading the '{}' file!".format(filepath))
        print(e)
        sys.exit()
    return file

import pandas as pd

def find_closest_zone(travel,zone_info,st_ind,other_zones):

    travel=np.delete(np.delete(travel,st_ind,0),st_ind,1)

    mask = np.logical_and(travel[:,zone_info.name] > 0.0, other_zones.values)
    #zoneindex=np.argmin(travel[zone_info.name,:][travel[zone_info.name,:]!=0.0]+travel[:,zone_info.name])
    subset=np.argmin(travel[zone_info.name,:][mask]+travel[:,zone_info.name][mask])
    zoneindex=np.arange(travel.shape[0])[mask][subset]
    return zoneindex

def route_data_parse(stops,travel_info):
    
        r_data=pd.json_normalize(stops,max_level=2)
        k=4
        r_chunks = pd.DataFrame(np.array([r_data.loc[0][i:i+k].values for i in range(0,r_data.shape[1],k)]),columns=['lat','lng','type','zone'])
        r_chunks=pd.concat([r_chunks,pd.DataFrame(stops.keys(),columns=['stop'])],axis=1)
        st_code=r_chunks[r_chunks.type=='Station'].stop.values[0]
        st_lat=r_chunks[r_chunks.type=='Station'].lat.values[0]
        st_lng=r_chunks[r_chunks.type=='Station'].lng.values[0]
        #st_zone=-1 #'ST' if r_chunks[r_chunks.type=='Station'].zone.isnull().values==True else r_chunks[r_chunks.type=='Station'].zone.values[0]
        index_names = r_chunks[r_chunks.type=='Station'].index
        r_chunks.drop(index_names, inplace = True)
        r_chunks=r_chunks.reset_index(drop=True)
        #SET ZONE ID USING HAVERSINE FORMULA
        
        while len(r_chunks[r_chunks.zone.isnull()==True]) > 0:
            zn_id=find_closest_zone(np.array(travel_info),r_chunks[r_chunks.zone.isnull()==True].iloc[0],index_names[0],r_chunks.zone.isnull()==False)
            #print(r_chunks.zone[zn_id])
            #zn_id=getDistanceFromLatLonInKm(r_chunks[r_chunks.zone.isnull()==True].iloc[0],r_chunks[r_chunks.zone.isnull()==False])
            up_ind=r_chunks[r_chunks.zone.isnull()==True].index[0]
            r_chunks.at[up_ind,'zone']=r_chunks.zone[zn_id]
        zone_enc= pd.DataFrame(r_chunks.zone.values,columns=['zone'])
        return index_names,zone_enc,st_code,st_lat,st_lng,r_chunks.lat,r_chunks.lng,r_chunks.stop
        
import datetime
from dateutil.parser import parse

def package_data(pd_data,d_time,d_date,k):
    pdj=pd.json_normalize(pd_data,max_level=3) 
    stop_ind=pdj.columns.str.split('.')
    #k=6
    stop_chunks=np.unique([st[0] for st in stop_ind]) #np.array([stop_ind[i] for i in range(0,stop_ind.shape[0],k)])
    pdj_chunks=pd.DataFrame(np.array([pdj.loc[0][i:i+k].values for i in range(0,pdj.shape[1],k)]))
    
    ## Total volume computation
    vol_chunks=pd.DataFrame([pdj_chunks[k-3].loc[i]*pdj_chunks[k-2].loc[i]*pdj_chunks[k-1].loc[i] for i in range(0,len(pdj_chunks[0]))],columns=['volume'])
    
    #Date and time parsing
    d_tm=datetime.datetime.combine(parse(d_date), parse(d_time).time())
    r_stm=pd.DataFrame([max((parse(st_time)-d_tm).total_seconds(),0) if isinstance(st_time, str) else 0 for st_time in pdj_chunks[1]],columns=['rel_start'])
    r_etm=pd.DataFrame([(parse(st_time)-d_tm).total_seconds() if isinstance(st_time, str) else 86400 for st_time in pdj_chunks[2]],columns=['rel_end'])
			# CHECK FOR OVERLAP
    pdj_chunks=pd.concat([r_stm,r_etm,pdj_chunks.rename(columns={3:'dur'}).dur,vol_chunks],axis=1)
    pdj_chunks=pd.concat([pdj_chunks, pd.DataFrame(stop_chunks,columns=['stop'])],axis=1)
    dur_sum=pdj_chunks.groupby(['stop']).dur.sum()
    stop_vol=pdj_chunks.groupby(['stop']).volume.sum()
    #stop_cap.append(stop_vol.tolist())
    min_time=pdj_chunks.groupby(['stop']).rel_start.max()
    max_time=pdj_chunks.groupby(['stop']).rel_end.min()
    pdj_chunks=pd.merge(stop_vol,pd.merge(pd.merge(dur_sum,min_time,on='stop'),max_time,on='stop'),on='stop').reset_index()
    #pdj_chunks=pd.merge(pdj_chunks,r_chunks,on='stop') 
    #package_info.append(pd.concat([],axis=1).values.tolist())
    #stop_zone.append(zone_enc.zone.values)
    #stop_info.append(pd.concat([r_chunks.lat,r_chunks.lng,r_chunks.stop,pdj_chunks.dur,pdj_chunks.rel_start,pdj_chunks.rel_end,pdj_chunks.volume,ac_seq[0]],axis=1).values.tolist())
    
    return pdj_chunks.dur,pdj_chunks.rel_start,pdj_chunks.rel_end,pdj_chunks.volume
from sklearn.preprocessing import OrdinalEncoder,LabelEncoder
scores=['High','Medium','Low']

def make_dataset_train(ac_data,routeinfo,mat,pd_data,rt_id):
    capacity=routeinfo['executor_capacity_cm3']
    d_date=routeinfo['date_YYYY_MM_DD']
    d_time=routeinfo['departure_time_utc']

    route_score=scores.index(routeinfo['route_score']) #changed
    
    stops=routeinfo['stops']
    
    travel_info=np.array([[mat[origin][destination] for destination in mat[origin]] for origin in mat])
    
    st_index,zone_enc,st_code,st_lat,st_lng,stop_lat,stop_lng,stop_seq=route_data_parse(stops,travel_info)
    depot=[st_lat,st_lng]

    st_times=np.concatenate((np.delete(travel_info[st_index,:],st_index) [:,None],np.delete(travel_info[:,st_index],st_index)[:,None]),axis=1)
    travel_info=np.delete(np.delete(travel_info,st_index,0),st_index,1)
    dur,start,end,vol=package_data(pd_data,d_time,d_date,7) #changed  

    ac_seq=pd.json_normalize(ac_data['actual'],max_level=0).transpose()
    ac_seq.drop(st_code, inplace = True)
    ac_seq=ac_seq.reset_index()
     #,int(route_score), #changes
    #return (torch.tensor([stop_lat,stop_lng], dtype=torch.float),rt_id,torch.tensor(vol, dtype=torch.float) / capacity,st_code,torch.tensor(ac_seq[0], dtype=torch.int),torch.tensor(depot, dtype=torch.float),stop_seq.tolist(),zone_enc.values.tolist(),torch.tensor(np.concatenate((dur.values[:,None],start.values[:,None],end.values[:,None]),1), dtype=torch.float),torch.tensor(travel_info,dtype=torch.float),torch.tensor(st_times,dtype=torch.float))
    
    return {
            'loc': torch.tensor([stop_lat,stop_lng], dtype=torch.float) ,
            'rt_id': rt_id,
            'demand': torch.tensor(vol, dtype=torch.float) / capacity,
            'st_code': st_code,
             'actual': torch.tensor(ac_seq[0], dtype=torch.int) ,
            'depot': torch.tensor(depot, dtype=torch.float) ,
            #'stop_seq':stop_seq.tolist(),
            'rt_score':int(route_score), #changes
            'zone':zone_enc.values.tolist(),
            'time_const':torch.tensor(np.concatenate((dur.values[:,None],start.values[:,None],end.values[:,None]),1).tolist(), dtype = torch.float),
            'travel_times':torch.tensor(travel_info,dtype=torch.float),
            'station_times':torch.tensor(st_times,dtype=torch.float)
            }
    
    

def make_dataset_test(routeinfo,mat,pd_data,rt_id):
    capacity=routeinfo['executor_capacity_cm3']
    d_date=routeinfo['date_YYYY_MM_DD']
    d_time=routeinfo['departure_time_utc']

    stops=routeinfo['stops']
    travel_info=np.array([[mat[origin][destination] for destination in mat[origin]] for origin in mat])
    
    st_index,zone_enc,st_code,st_lat,st_lng,stop_lat,stop_lng,stop_seq=route_data_parse(stops,travel_info)
    depot=[st_lat,st_lng]

    st_times=np.concatenate((np.delete(travel_info[st_index,:],st_index) [:,None],np.delete(travel_info[:,st_index],st_index)[:,None]),axis=1)
    travel_info=np.delete(np.delete(travel_info,st_index,0),st_index,1)
    dur,start,end,vol=package_data(pd_data,d_time,d_date,6)  


    return {
            'loc': torch.tensor([stop_lat,stop_lng], dtype=torch.float) ,
            'rt_id': rt_id,
            'demand': torch.tensor(vol, dtype=torch.float) / capacity,
            'st_code': st_code,
            'depot': torch.tensor(depot, dtype=torch.float) ,
            'stop_seq':stop_seq.tolist(),
            'zone':zone_enc.values,
            'time_const':torch.tensor(np.concatenate((dur.values[:,None],start.values[:,None],end.values[:,None]),1).tolist(), dtype=torch.float),
            'travel_times':torch.tensor(travel_info,dtype=torch.float),
            'station_times':torch.tensor(st_times,dtype=torch.float)
            }
    

    
from itertools import chain
class VRPDataset(Dataset):
    
    def __init__(self,distribution='train'):
        super(VRPDataset, self).__init__()
        
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if distribution == 'train':
            #changed
             
            actual_routes_json = os.path.join(BASE_DIR,'../data/model_build_inputs/actual_sequences.json')
            print(actual_routes_json)
            invalid_scores_json = os.path.join(BASE_DIR,'../data/model_build_inputs/invalid_sequence_scores.json')
            rt_data_path=os.path.join(BASE_DIR, '../data/model_build_inputs/route_data.json')
            pack_data_path= os.path.join(BASE_DIR, '../data/model_build_inputs/package_data.json')
            cost_matrices_json = os.path.join(BASE_DIR,'../data/model_build_inputs/travel_times.json')
            output_path=os.path.join(BASE_DIR, '../data/model_build_outputs/classes.npy')
            
            cost_matrices=read_json_data(cost_matrices_json)
            invalid_scores=read_json_data(invalid_scores_json)
            print('read_matrices') 
            actual_routes=read_json_data(actual_routes_json)
            print('Done')
            rt_data=read_json_data(rt_data_path)
            print('Done')
            pack_data=read_json_data(pack_data_path)
            print('read_all data')
            self.data = [make_dataset_train(actual_routes[route],rt_data[route],cost_matrices[route],pack_data[route],route) for route in rt_data]
                
            # Encoding of scores and zones
            le=LabelEncoder()
            zone_set=list(set(chain(*[list(chain(*dt['zone'])) for dt in self.data])))
            le.fit(zone_set)
            #st=[np.array(self.le.transform(dt['zone']),dtype=int) for dt in self.data]
            for i,dt in enumerate(self.data):
                self.data[i]['zone']=torch.cat((torch.tensor([0.0]),torch.tensor(le.transform(dt['zone'])+1,dtype=torch.float)),0)
            np.save(output_path, le.classes_)
            

            
        elif distribution == 'val' or distribution == 'test':
            rt_data_path=os.path.join(BASE_DIR, '../data/model_apply_inputs/new_route_data.json')
            pack_data_path= os.path.join(BASE_DIR, '../data/model_apply_inputs/new_package_data.json')
            cost_matrices_json = os.path.join(BASE_DIR,'../data/model_apply_inputs/new_travel_times.json')
            
            output_path=os.path.join(BASE_DIR, '../data/model_build_outputs/classes.npy')
            
            rt_data=read_json_data(rt_data_path)
            pack_data=read_json_data(pack_data_path)
            cost_matrices=read_json_data(cost_matrices_json)
            #invalid_scores=read_json_data(invalid_scores_json)
            
            self.data = [make_dataset_test(rt_data[route],cost_matrices[route],pack_data[route],route) for route in rt_data]

            #Encodeing of zones
            import bisect
            le = LabelEncoder()
            le.classes_= np.load(output_path)
            le_classes = le.classes_.tolist()
            zone_set=list(set(chain(*[list(chain(*dt['zone'])) for dt in self.data])))
            temp_zones=[bisect.insort_left(le_classes, zone) for zone in zone_set if (zone not in le_classes)]
            le.classes_=le_classes
            
            for i,dt in enumerate(self.data):
                self.data[i]['zone']=torch.cat((torch.tensor([0.0]),torch.tensor(le.transform(dt['zone'])+1,dtype=torch.float)),0)
            
        self.size=len(self.data)
    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]
