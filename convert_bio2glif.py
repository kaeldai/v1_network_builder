import os
import shutil
import pandas as pd
import numpy as np
from pprint import pprint
import h5py
import argparse
import json

from bmtk.utils.sonata import File


def copy_nodes(src_network, bio_network_dir='network_bio', glif_network_dir='network_glif'):
    bio_nodes_path = os.path.join(bio_network_dir, '{}_nodes.h5'.format(src_network))
    bio_nodes_types_path = os.path.join(bio_network_dir, '{}_node_types.csv'.format(src_network))

    glif_nodes_path = os.path.join(glif_network_dir, '{}_nodes.h5'.format(src_network))
    glif_node_types_path = os.path.join(glif_network_dir, '{}_node_types.csv'.format(src_network))

    print('> Copying {} to {}'.format(bio_nodes_path, glif_nodes_path))
    shutil.copy2(bio_nodes_path, glif_nodes_path)
    print('>   done')

    print('> Copying {} to {}'.format(bio_nodes_types_path, glif_node_types_path))
    shutil.copy2(bio_nodes_types_path, glif_node_types_path)
    print('>   done')


def convert_v1_nodes(bio_network_dir='network_bio', glif_network_dir='network_glif'):
    v1_models_glif = json.load(open('glif_props/v1_node_models.json', 'r'))
    bio2glif_map = json.load(open('glif_props/bio2glif_type_map.json', 'r'))
    bio2glif_map = {int(k): v for k, v in bio2glif_map.items()}

    glif_ntid_lu = {}
    for loc, loc_dict in v1_models_glif['locations'].items():
        for pop_name, pop_dict in loc_dict.items():
            for model_props in pop_dict['models']:
                glif_ntid_lu[model_props['node_type_id']] = model_props

    sonata = File(
        data_files=os.path.join(bio_network_dir, 'v1_nodes.h5'),
        data_type_files=os.path.join(bio_network_dir, 'v1_node_types.csv')
    )

    nodes_df = sonata.nodes['v1'].to_dataframe()
    nodes_df['glif_node_type_id'] = 0

    bio_v1_nodes_path = os.path.join(bio_network_dir, 'v1_nodes.h5')
    glif_v1_nodes_path = os.path.join(glif_network_dir, 'v1_nodes.h5')
    glif_v1_node_types_path = os.path.join(glif_network_dir, 'v1_node_types.csv')

    if not os.path.exists(glif_network_dir):
        os.mkdir(glif_network_dir)

    if os.path.exists(os.path.join(glif_network_dir, 'v1_nodes.h5')):
        pass

    print('> Copying {} to {}'.format(bio_v1_nodes_path, glif_v1_nodes_path))
    shutil.copy2(bio_v1_nodes_path, glif_v1_nodes_path)
    print('>   done')

    # convert the bio node_type_ids to
    print('> Creating {}'.format(glif_v1_nodes_path))
    for bio_ntid, subgrp in nodes_df.groupby('node_type_id'):
        assert(bio_ntid in bio2glif_map)
        # print(bio_ntid)
        glif_props = bio2glif_map[bio_ntid]
        glif_ntids = glif_props['node_type_id']
        if len(glif_ntids) == 1:
            # Convert each bio node-type model to the glif version
            nodes_df.loc[subgrp.index.values, 'glif_node_type_id'] = glif_ntids
        else:
            # Each biophys LIF* nodes can be converted into multiple models, selected randomly
            randomized_ntids = np.random.choice(glif_ntids, size=len(subgrp), replace=True, p=glif_props['proportions'])
            nodes_df.loc[subgrp.index.values, 'glif_node_type_id'] = randomized_ntids

    # save v1_nodes.h5 for GLIF, copy the biophys nodes.h5 directly and update the node_type_id
    with h5py.File(glif_v1_nodes_path, 'a') as h5:
        del h5['/nodes/v1/node_type_id']
        h5['/nodes/v1'].create_dataset('node_type_id', data=nodes_df['glif_node_type_id'].values)
    print('>   done')

    # create and v1_node_types.csv table
    print('> Creating {}'.format(glif_v1_node_types_path))
    node_types_table = []
    node_type_cols = ['node_type_id', 'model_type', 'model_template', 'dynamics_params']
    valid_node_ids = np.unique(nodes_df['glif_node_type_id']).tolist()  # use only node-types found in h5
    for loc, loc_dict in v1_models_glif['locations'].items():
        for pop_name, pop_dict in loc_dict.items():
            for model_props in pop_dict['models']:
                if model_props['node_type_id'] in valid_node_ids:
                    tmp_props = {k: v for k, v in model_props.items() if k in node_type_cols}
                    tmp_props.update({'pop_name': pop_name, 'location': loc, 'population': 'v1'})
                    node_types_table.append(tmp_props)

    node_types_table_df = pd.DataFrame(node_types_table)
    col_order = node_type_cols + [c for c in node_types_table_df.columns.tolist() if c not in node_type_cols]
    node_types_table_df[col_order].to_csv(glif_v1_node_types_path, sep=' ', index=False)
    print('>   done.')


def convert_edges(src_network, bio_network_dir='network_bio', glif_network_dir='network_glif', block_size=10000000,
                  join_on_src=True):
    bio_edges_path = os.path.join(bio_network_dir, '{}_v1_edges.h5'.format(src_network))
    bio_edge_types_path = os.path.join(bio_network_dir, '{}_v1_edge_types.csv'.format(src_network))

    glif_edges_path = os.path.join(glif_network_dir, '{}_v1_edges.h5'.format(src_network))
    glif_edge_types_path = os.path.join(glif_network_dir, '{}_v1_edge_types.csv'.format(src_network))

    edge_population = '/edges/{}_to_v1'.format(src_network)

    sonata = File(
        data_files=os.path.join(glif_network_dir, 'v1_nodes.h5'.format(src_network)),
        data_type_files=os.path.join(glif_network_dir, 'v1_node_types.csv'.format(src_network))
    )
    node_types_lu = sonata.nodes['v1'].to_dataframe()[['node_type_id', 'pop_name']]

    bio_edges_h5 = h5py.File(bio_edges_path, 'r')
    bio_edges_grp = bio_edges_h5[edge_population]
    n_edges = len(bio_edges_grp['edge_type_id'])

    block_ranges = list(range(0, n_edges, block_size)) + [n_edges]
    block_steps = [(block_ranges[i], block_ranges[i+1]) for i in range(len(block_ranges)-1)]

    glif_edge_types_df = pd.read_csv(os.path.join('glif_props', '{}_v1_edge_types.csv'.format(src_network)), sep=' ')
    def convert_query(s):
        query_val = s.split('==')[-1]
        query_val = query_val.replace("'", '')
        return query_val

    edges_table_lu = pd.DataFrame({
        'glif_edge_type_id': glif_edge_types_df['edge_type_id'],
        'target_node_type_id': glif_edge_types_df['target_query'].apply(convert_query).astype(np.int64),
        'source_pop_name': glif_edge_types_df['source_query'].apply(convert_query)
    })

    print('> Copying {} to {}'.format(bio_edges_path, glif_edges_path))
    shutil.copy2(bio_edges_path, glif_edges_path)
    print('>   done.')

    print('> Updating edge_type_id in {}. {} edges to convert'.format(glif_edges_path, n_edges))
    join_cols = ['target_node_type_id', 'source_pop_name'] if join_on_src else ['target_node_type_id']
    with h5py.File(glif_edges_path, 'a') as h5:
        del h5[edge_population]['edge_type_id']
        h5[edge_population].create_dataset('edge_type_id', shape=(n_edges, ), dtype=np.int)

        valid_edge_type_ids = set()
        for idx_beg, idx_end in block_steps:
            print('>   block [{}, {})'.format(idx_beg, idx_end))
            edges_df = pd.DataFrame({
                'source_node_id': bio_edges_grp['source_node_id'][idx_beg:idx_end],
                'target_node_id': bio_edges_grp['target_node_id'][idx_beg:idx_end]
            })

            edges_df = edges_df.merge(node_types_lu[['node_type_id']], how='left', left_on='target_node_id', right_index=True)
            edges_df = edges_df.merge(node_types_lu[['pop_name']], how='left', left_on='source_node_id', right_index=True)
            edges_df = edges_df.rename(columns={'node_type_id': 'target_node_type_id', 'pop_name': 'source_pop_name'})

            merged_table = edges_df.merge(edges_table_lu, how='left', on=join_cols)
            h5[edge_population]['edge_type_id'][idx_beg:idx_end] = merged_table['glif_edge_type_id']
            valid_edge_type_ids |= set(merged_table['glif_edge_type_id'].unique().tolist())

    print('>   done.')

    print('> Saving {}.'.format(glif_edge_types_path))
    glif_edge_types_df[
        glif_edge_types_df['edge_type_id'].isin(list(valid_edge_type_ids))
    ].to_csv(glif_edge_types_path, sep=' ', index=False)
    print('>   done.')


def create_bkg_table(glif_network_dir='network_glif'):
    v1_node_types_df = pd.read_csv(os.path.join(glif_network_dir, 'v1_node_types.csv'), sep=' ')[['node_type_id', 'pop_name']]

    bkg_edge_types_table = pd.read_csv('/data/work_files/V1_network_update/Glif_network/network/bkg_v1_edge_types.csv', sep=' ')
    bkg_edge_types_table['pop_name'] = bkg_edge_types_table.apply(
        lambda r: r['target_query'].split('==')[-1].replace("'", ''),
        axis=1
    )

    merged_data = v1_node_types_df.merge(bkg_edge_types_table, how='left', on='pop_name')
    merged_data['target_query'] = merged_data.apply(
        lambda r: "node_type_id=='{}'".format(r['node_type_id']),
        axis=1
    )
    merged_data['edge_type_id'] = range(100, 100+len(merged_data))
    merged_data = merged_data.drop(labels=['node_type_id', 'pop_name'], axis=1)

    merged_data.to_csv('glif_props/bkg_v1_edge_types.csv', sep=' ', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert V1 Biophysical to GLIF network model')
    parser.add_argument('-b', '--biophys-dir', default='network', help='directory containing Bio V1 network.')
    parser.add_argument('-g', '--glif-dir', default='network_glif', help='directory containing Bio V1 network.')
    parser.add_argument('networks', type=str, nargs='*', default=['v1', 'bkg', 'lgn'])
    args = parser.parse_args()
    nets = set(args.networks)

    if not os.path.exists(args.glif_dir):
        os.mkdir(args.glif_dir)

    if 'v1' in nets:
        convert_v1_nodes(bio_network_dir=args.biophys_dir, glif_network_dir=args.glif_dir)
        convert_edges('v1', bio_network_dir=args.biophys_dir, glif_network_dir=args.glif_dir)

    if 'lgn' in nets:
        copy_nodes('lgn', bio_network_dir=args.biophys_dir, glif_network_dir=args.glif_dir)
        convert_edges('lgn', bio_network_dir=args.biophys_dir, glif_network_dir=args.glif_dir, join_on_src=False)

    if 'bkg' in nets:
        copy_nodes('bkg', bio_network_dir=args.biophys_dir, glif_network_dir=args.glif_dir)
        convert_edges('bkg', bio_network_dir=args.biophys_dir, glif_network_dir=args.glif_dir, join_on_src=False)
