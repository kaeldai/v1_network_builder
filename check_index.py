from bmtk.utils import sonata
from bmtk.utils.reports import SpikeTrains
import numpy as np
import h5py
import pandas as pd

# from memory_profiler import memory_usage


def iter_edges():
    net = sonata.File(
        data_files='network_shuffled/lgn_v1_edges.shuffled.00per_changed.h5',
        data_type_files='network_shuffled/lgn_v1_edge_types.csv'
    )

    # net = sonata.File(
    #     data_files='network_orig/lgn_v1_edges.h5',
    #     data_type_files='network_orig/lgn_v1_edge_types.csv'
    # )

    edges = net.edges['lgn_to_v1']

    spikes = SpikeTrains.load(
        path='full3_GScorrected_PScorrected_3.0sec_SF0.04_TF2.0_ori0.0_c100.0_gs0.5_spikes.trial_0.h5',
        file_type='h5'
    )

    # for node_id in range(17500):
    #     spike_times = spikes.get_times(node_id=node_id)
    #     print(spike_times)

    n_edges = 0
    for trg_nid in range(0, 230924, 4618):
        # if trg_nid % 1000 == 0:
        #     print(trg_nid)

        for i, edge in enumerate(edges.get_target(trg_nid)):
            spike_times = spikes.get_times(node_id=edge.source_node_id)
            # print(edge.source_node_id)
            n_edges += 1

    print(n_edges)


def check_tmp_edges(edges_path='/data/Workspace/v1_network_builder/network_orig/lgn_v1_edges.h5'):
    edges_orig_h5 = h5py.File(edges_path, 'r')
    edges_orig_df = pd.DataFrame({
        'edge_type_id': edges_orig_h5['/processed/edge_type_id'][()],
        'nsyns': edges_orig_h5['/processed/0/nsyns'][()]
    })

    for etid, nsyns_df in edges_orig_df.groupby('edge_type_id'):
        print(etid, nsyns_df['nsyns'].unique())


def check_nsyns(edges_path):
    edges_orig_h5 = h5py.File(edges_path, 'r')
    edges_orig_df = pd.DataFrame({
        'edge_type_id': edges_orig_h5['/edges/lgn_to_v1/edge_type_id'][()],
        'nsyns': edges_orig_h5['/edges/lgn_to_v1/0/nsyns'][()]
    })

    for etid, nsyns_df in edges_orig_df.groupby('edge_type_id'):
        print(etid, nsyns_df['nsyns'].unique())


def cmp_edges(edges_orig, edges_tmp):
    edges_orig_h5 = h5py.File(edges_orig, 'r')
    edges_orig_df = pd.DataFrame({
        'edge_type_id': edges_orig_h5['/edges/lgn_to_v1/edge_type_id'][()],
        'nsyns': edges_orig_h5['/edges/lgn_to_v1/0/nsyns'][()]
    })
    nsyns_orig = {}
    for etid, nsyns_df in edges_orig_df.groupby('edge_type_id'):
        nsyns = nsyns_df['nsyns'].unique()
        assert(len(nsyns) == 1)
        nsyns_orig[etid] = nsyns[0]
        # print(etid, nsyns_df['nsyns'].unique())

    edges_tmp_h5 = h5py.File(edges_tmp, 'r')
    # edges_tmp_df = pd.DataFrame({
    #     'edge_type_id': edges_tmp_h5['/processed/edge_type_id'][()],
    #     'nsyns': edges_tmp_h5['/processed/0/nsyns'][()]
    # })
    edges_tmp_df = pd.DataFrame({
        'edge_type_id': edges_tmp_h5['/edges/lgn_to_v1/edge_type_id'][()],
        'nsyns': edges_tmp_h5['/edges/lgn_to_v1/0/nsyns'][()]
    })

    for etid, nsyns_df in edges_orig_df.groupby('edge_type_id'):
        nsyns = nsyns_df['nsyns'].unique()
        if len(nsyns) > 1:
            print('-', etid, nsyns_df['nsyns'].unique())
        elif etid in nsyns_orig:
            print('+', etid, nsyns[0] == nsyns_orig[etid])


def check_edges(edges_path='network_rebuilt/lgn_v1_edges.h5'):
    edges_h5 = h5py.File(edges_path, 'r')
    edges_df = pd.DataFrame({
        'edge_type_id': edges_h5['/edges/lgn_to_v1/edge_type_id'][()],
        'nsyns': edges_h5['/edges/lgn_to_v1/0/nsyns'][()]
    })
    print(edges_df[edges_df['edge_type_id'] == 6591])

    # for etid, nsyns_df in edges_orig_df.groupby('edge_type_id'):
    #     print(etid, nsyns_df['nsyns'].unique())



if __name__ == '__main__':
    # mem = memory_usage((iter_edges))
    # print(np.max(mem))

    # exit()

    # check_nsyns(edges_path='network_orig/lgn_v1_edges.h5')
    print('----')
    check_nsyns(edges_path='network_rebuilt/lgn_v1_edges.h5')

    # check_edges(edges_path='network_rebuilt/.unsorted.lgn_v1_edges.h5')



    # check_nsyns(edge_path='/data/Workspace/v1_network_builder/network_rebuilt/lgn_v1_edges.h5')

    # check_tmp_edges(edges_path='.edge_types_table.processed.1.h5')

    cmp_edges(edges_orig='network_orig/lgn_v1_edges.h5', edges_tmp='network_rebuilt/lgn_v1_edges.h5')

