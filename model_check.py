import os
import sys
import numpy as np
import pandas as pd
import h5py
import argparse
import scipy.stats

from bmtk.utils import sonata


max_load_size = 10000000

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('expand_frame_repr', False)


def validate_nodes(nodes_path, node_types_path, population='v1'):
    # nodes_path = os.path.join(network_dir, '{}_nodes.h5'.format(population))
    nodes_h5 = h5py.File(nodes_path, 'r')

    # node_types_path = os.path.join(network_dir, '{}_node_types.csv'.format(population))
    node_types_df = pd.read_csv(node_types_path, sep=' ')

    pop_grp = nodes_h5['/nodes'][population]
    nodes_df = pd.DataFrame({
        'node_id': pop_grp['node_id'][()],
        'node_type_id': pop_grp['node_type_id'][()],
        'group_id': pop_grp['node_group_id'][()],
        'group_index': pop_grp['node_group_index'][()]
    })

    print('Checking {} nodes'.format(population))
    print('------------------------------------')
    print('   checking node_id (no repeats): {}'.format('✓' if nodes_df['node_id'].nunique() == len(nodes_df) else '✗'))
    print('    checking node_id (is sorted): {}'.format('✓' if np.all(np.diff(nodes_df['node_id']) > 0) else '✗'))
    print(' checking node_id (is contigous): {}'.format('✓' if np.all(np.diff(nodes_df['node_id']) == 1) else '✗'))

    # Check node_type_id in hdf5 can be found in the node_types.csv file
    h5_node_type_ids = set(nodes_df['node_type_id'])
    csv_node_type_ids = set(node_types_df['node_type_id'])
    unmatched_ids = h5_node_type_ids - csv_node_type_ids
    print('           checking node_type_id: {}'.format('✓' if not unmatched_ids else '✗'))

    # Check node_group_id and node_group_index
    for grp_id, indx_subset in nodes_df.groupby('group_id'):
        if str(grp_id) not in pop_grp:
            print('          checking node_group_id: ✗ (could not find model group {})'.format(grp_id))
            break

        subgrp = pop_grp[str(grp_id)]
        if len(subgrp.items()) > 0:
            max_idx = indx_subset['group_index'].max()
            for params_name, params_ds in subgrp.items():
                if max_idx >= len(params_ds):
                    print('       checking node_group_index: ✗ (group {}/{} contains index out-of-range)'.format(grp_id, params_name))
                    break

    else:
        print('          checking node_group_id: ✓')
        print('       checking node_group_index: ✓')

    n_unique_ids = nodes_df['node_id'].nunique()
    print('                     total nodes: {:,}'.format(n_unique_ids))

    n_node_types = nodes_df['node_type_id'].nunique()
    print('                total node_types: {:,}'.format(n_node_types))
    print('------------------------------------')


def get_node_stats(nodes_path, node_types_path, population, group_columns='pop_name', sort_by=None, save_as=None):
    group_columns = group_columns if isinstance(group_columns, list) else [group_columns]
    # nodes_path = os.path.join(network_dir, '{}_nodes.h5'.format(population))
    # node_types_path = os.path.join(network_dir, '{}_node_types.csv'.format(population))

    net = sonata.File(
        data_files=nodes_path,
        data_type_files=node_types_path
    )

    nodes_df = net.nodes[population].to_dataframe(index_by_id=False)
    n_ids = len(nodes_df)

    node_counts_df = nodes_df.groupby(group_columns).size().to_frame(name='counts')
    node_counts_df['percents'] = node_counts_df.apply(lambda r: float(r['counts']) / n_ids*100.0, axis=1)
    node_counts_df = node_counts_df.reset_index()

    coord_columns = [c for c in ['x', 'y', 'z'] if c in nodes_df.columns]
    if coord_columns:
        coords_stats_df = nodes_df[group_columns + coord_columns].groupby(group_columns).agg([np.mean, np.std, 'min', 'max'])
        coords_stats_df.columns = ['_'.join(col).strip() for col in coords_stats_df.columns.values]
        coords_stats_df = coords_stats_df.reset_index()
        node_counts_df = node_counts_df.merge(coords_stats_df, how='left', on=group_columns)

    if sort_by:
        node_counts_df = node_counts_df.sort_values(sort_by)

    if save_as:
        node_counts_df.to_csv(save_as, sep=' ', index=False)
    else:
        print(node_counts_df)


def cmp_node_types(orig_nodes_path, orig_node_types_path, new_nodes_path, new_node_types_path, population='v1',
                   grp_cols=['dynamics_params', 'morphology', 'pop_name', 'model_type', 'model_template'],
                   check_coords=False):
    title_str = 'comparing {} (new) with {} (orig) using columns: {}'.format(new_nodes_path, orig_nodes_path, ', '.join(grp_cols))
    print(title_str)
    print('-'*len(title_str))

    net = sonata.File(
        data_files=orig_nodes_path,
        data_type_files=orig_node_types_path
    )
    orig_nodes_df = net.nodes[population].to_dataframe(index_by_id=False)
    orig_nodes_df[grp_cols] = orig_nodes_df[grp_cols].fillna('NULL')

    net = sonata.File(
        data_files=new_nodes_path,
        data_type_files=new_node_types_path
    )
    rebuilt_nodes_df = net.nodes[population].to_dataframe(index_by_id=False)
    rebuilt_nodes_df[grp_cols] = rebuilt_nodes_df[grp_cols].fillna('NULL')

    # Count the number of nodes in each group and compare original to new
    orig_node_type_counts = orig_nodes_df[grp_cols].groupby(grp_cols).size().reset_index(name='counts')
    rebuilt_node_type_counts = rebuilt_nodes_df[grp_cols].groupby(grp_cols).size().reset_index(name='counts')
    cmp_nodes_df = orig_node_type_counts.merge(rebuilt_node_type_counts, how='outer', on=grp_cols,
                                               suffixes=['_orig', '_new'])
    cmp_nodes_df['counts_orig'] = cmp_nodes_df['counts_orig'].fillna(0).astype(np.int)
    cmp_nodes_df['counts_new'] = cmp_nodes_df['counts_new'].fillna(0).astype(np.int)
    cmp_nodes_df['count_diffs'] = np.abs(cmp_nodes_df['counts_new'] - cmp_nodes_df['counts_orig'])
    count_diffs = cmp_nodes_df['count_diffs'].values
    if np.all(count_diffs == 0):
        print('node_types model counts match: ✓')
    else:
        print('node_types model counts match: ✗, showing differences below:')
        print(cmp_nodes_df[cmp_nodes_df['count_diffs'] != 0])

    # check coordinates
    coord_columns = [c for c in ['x', 'y', 'z'] if c in orig_nodes_df.columns]
    if check_coords and coord_columns:
        coords_orig_df = orig_nodes_df[grp_cols + coord_columns].groupby(grp_cols).agg([np.mean, 'count']) # , 'min', 'max'])
        coords_orig_df.columns = ['_'.join(col).strip() for col in coords_orig_df.columns.values]
        coords_orig_df['node_count'] = coords_orig_df['{}_count'.format(coord_columns[0])]
        coords_orig_df = coords_orig_df.drop(columns=['{}_count'.format(c) for c in coord_columns])

        coords_new_df = rebuilt_nodes_df[grp_cols + coord_columns].groupby(grp_cols).agg([np.mean]) # , 'min', 'max'])
        coords_new_df.columns = ['_'.join(col).strip() for col in coords_new_df.columns.values]

        merged_coords_df = coords_orig_df.merge(coords_new_df, on=grp_cols, suffixes=['_orig', '_new'], how='outer', indicator=True)
        merged_coords_df = merged_coords_df[merged_coords_df['_merge'] == 'both']
        merged_coords_df['center_diffs'] = np.sqrt(np.sum((merged_coords_df['{}_mean_orig'.format(c)] - merged_coords_df['{}_mean_new'.format(c)])**2 for c in coord_columns))


def validate_edges(edges_path, edge_types_path, trg_nodes_path, src_nodes_path=None, models_dir=None):
    np.set_printoptions(threshold=sys.maxsize)
    edge_types_df = pd.read_csv(edge_types_path, sep=' ', index_col=False)

    def _check_col_exists(col_name):
        mark = '✓' if col_name in edge_types_df else '✗'
        print('  {:>30}: {}'.format(col_name, mark))

    def _check_syn_models_exists():
        dyn_params = edge_types_df['dynamics_params'].unique()
        dyn_params_exists = [os.path.exists(os.path.join(models_dir, 'synaptic_models', dyn_param))
                             for dyn_param in dyn_params]
        if all(dyn_params_exists):
            print('  has dynamic_params: ✓')
        else:
            missing_indices = np.argwhere(~np.array(dyn_params_exists)).flatten()
            print('  has dynamic_params: ✗, missing {}'.format(dyn_params[missing_indices]))

    def _check_ids_contigous():
        edge_types_ids = edge_types_df['edge_type_id'].values
        et_ids_diffs = np.diff(edge_types_ids)
        mark = '✓' if np.all(et_ids_diffs == 1) else '✗'
        print(' '*18 + 'contiguous ids: {}'.format(mark))

    header_str = 'Checking edge_types file "{}"'.format(edge_types_path)
    print(header_str)
    print('-'*len(header_str))

    _check_col_exists('model_template')
    _check_col_exists('dynamics_params')
    _check_col_exists('syn_weight')
    _check_col_exists('delay')
    if models_dir:
        _check_syn_models_exists()
    _check_ids_contigous()
    edge_types_ids = edge_types_df['edge_type_id'].values
    min_et_id = np.min(edge_types_ids)
    max_et_id = np.max(edge_types_ids)

    with h5py.File(trg_nodes_path, 'r') as nodes_h5:
        assert(len(nodes_h5['/nodes'].keys()) == 1)
        target_population = list(nodes_h5['/nodes'].keys())[0]
        target_node_ids = np.array(nodes_h5['/nodes'][target_population]['node_id'])
        min_target_id = np.min(target_node_ids)
        max_target_id = np.max(target_node_ids)

    if src_nodes_path is None:
        source_population = target_population
        min_source_id = min_target_id
        max_source_id = max_target_id
    else:
        with h5py.File(src_nodes_path, 'r') as nodes_h5:
            assert(len(nodes_h5['/nodes'].keys()) == 1)
            source_population = list(nodes_h5['/nodes'].keys())[0]
            source_node_ids = np.array(nodes_h5['/nodes'][source_population]['node_id'])
            min_source_id = np.min(source_node_ids)
            max_source_id = np.max(source_node_ids)

    # Check edges
    edges_h5 = h5py.File(edges_path, 'r')

    def _has_ds(ds_name, grp):
        mark = '✓' if ds_name in grp else '✗'
        print('  {:>30}: {}'.format('contains column ' + ds_name, mark))

    def _check_ds_values(edges_grp, ds_name, min_id, max_id):
        is_sorted = True
        are_ids_valid = True
        unique_vals = np.array([])

        grp = edges_grp[ds_name]
        n_edges = grp.shape[0]
        chunk_idx_beg = 0
        chunk_idx_end = int(np.min((max_load_size, n_edges)))
        global_min = n_edges
        global_max = 0
        while chunk_idx_beg < n_edges:
            vals = np.array(grp[chunk_idx_beg:chunk_idx_end])
            min_v = np.min(vals)
            max_v = np.max(vals)
            are_ids_valid = are_ids_valid and (min_v >= min_id or max_v <= max_id)

            is_sorted = is_sorted and np.all(np.diff(vals) >= 0)
            if chunk_idx_end + 1 < n_edges:
                # Need to check at the ends of each chunk
                is_sorted = is_sorted and grp[chunk_idx_end + 1] >= grp[chunk_idx_end]

            unique_vals = np.unique(np.concatenate((unique_vals, np.unique(vals))))

            chunk_idx_beg = chunk_idx_end
            chunk_idx_end = np.min((chunk_idx_end + max_load_size, n_edges))
            global_min = np.min((min_v, global_min))
            global_max = np.max((max_v, global_max))

        print('  {:>16} has valid ids: {} [{:,} - {:,}]'.format(ds_name, '✓' if are_ids_valid else '✗',
                                                                global_min, global_max))
        print('  {:>30}: {}'.format('sorted by ' + ds_name, '✓' if is_sorted else '✗'))
        print('  {:>16} unique values: {:,}'.format(ds_name, len(unique_vals)))

    indices_lu = {
        'target_to_source': 'target_node_id',
        'source_to_target': 'source_node_id',
        'edge_type_to_index': 'edge_type_id'
    }

    def _check_indices(edges_grp, index_name):
        if 'indices' in edges_grp:
            indx_grp = edges_grp['indices']
        elif 'indicies' in edges_grp:
            indx_grp = edges_grp['indicies']
        else:
            print('  {:>24} index: ✗'.format(index_name))
            return

        has_index = index_name in indx_grp

        print('  {:>24} index: {}'.format(index_name,  '✓' if index_name in indx_grp else '✗'))
        if has_index:
            id_col = indices_lu[index_name]
            index_ids = edges_grp[id_col][()]
            # print(index_ids)
            # exit()
            # print(np.max(edges_grp[id_col]))
            # print(indx_grp[index_name]['node_id_to_range'].shape)

    print()
    header_str = 'Checking edges file "{}"'.format(edges_path)
    print(header_str)
    print('-'*len(header_str))
    for edges_pop, edges_grp in edges_h5['/edges'].items():
        _has_ds('source_node_id', edges_grp)
        _has_ds('target_node_id', edges_grp)
        _has_ds('edge_type_id', edges_grp)
        n_srcs = edges_grp['source_node_id'].shape[0]
        n_trgs = edges_grp['source_node_id'].shape[0]
        n_ets = edges_grp['edge_type_id'].shape[0]
        print('{:>32}: {} ({:,} rows)'.format('dimensions match', '✓' if n_srcs == n_trgs == n_ets else '✗', n_srcs))
        _check_ds_values(edges_grp, 'source_node_id', min_source_id, max_source_id)
        _check_ds_values(edges_grp, 'target_node_id', min_target_id, max_target_id)
        _check_ds_values(edges_grp, 'edge_type_id', min_et_id, max_et_id)
        _check_indices(edges_grp, 'edge_type_to_index')
        _check_indices(edges_grp, 'source_to_target')
        _check_indices(edges_grp, 'target_to_source')


def check_edges_for_lgn(edges_files, edge_types_file, edges_pop, trg_nodes_file, trg_node_types_file, trg_nodes_pop,
                        src_nodes_file, src_node_types_file, src_nodes_pop):
    trg_net = sonata.File(data_files=trg_nodes_file, data_type_files=trg_node_types_file)
    trg_nodes_df = trg_net.nodes[trg_nodes_pop].to_dataframe(index_by_id=False)
    trg_nodes_df = trg_nodes_df[['node_id', 'pop_name']]
    trg_nodes_df = trg_nodes_df.rename(columns={'node_id': 'target_node_id', 'pop_name': 'target_pop_name'})

    src_net = trg_net if src_nodes_file is None else sonata.File(data_files=src_nodes_file, data_type_files=src_node_types_file)
    src_nodes_df = src_net.nodes[src_nodes_pop].to_dataframe(index_by_id=False)
    src_nodes_df = src_nodes_df[['node_id', 'pop_name']]
    src_nodes_df = src_nodes_df.rename(columns={'node_id': 'source_node_id', 'pop_name': 'source_pop_name'})

    edges_h5 = h5py.File(edges_files, 'r')
    edge_types_df = pd.read_csv(edge_types_file, sep=' ', index_col=False)

    edges_grp = edges_h5['/edges'][edges_pop]
    edges_df = pd.DataFrame({
        'source_node_id': edges_grp['source_node_id'][()],
        'target_node_id': edges_grp['target_node_id'][()]
    })

    edges_df = edges_df.merge(trg_nodes_df, how='left', on='target_node_id')
    edges_df = edges_df.merge(src_nodes_df, how='left', on='source_node_id')

    print(edges_df['source_pop_name'].unique())


def get_edge_stats_by_ids(edges_path, edge_types_path, save_as=None):
    edge_types_df = pd.read_csv(edge_types_path, sep=' ', index_col=False)
    edge_types_df = edge_types_df[['edge_type_id', 'source_query', 'target_query']]
    edge_types_df['target_query'] = edge_types_df.apply(lambda r: r['target_query'].split('&')[0], axis=1)

    edges_h5 = h5py.File(edges_path, 'r')
    for edges_pop, edges_grp in edges_h5['/edges'].items():
        edges_df = pd.DataFrame({
            'edge_type_id': edges_grp['edge_type_id'],
            'edge_group_id': edges_grp['edge_group_id'],
            'edge_group_index': edges_grp['edge_group_index']
        })

        nsyns_df = None
        for group_id in np.unique(edges_grp['edge_group_id']):
            model_grp = edges_grp[str(group_id)]
            if 'nsyns' not in model_grp:
                continue

            grp_df = pd.DataFrame({
                'edge_group_id': group_id,
                'edge_group_index': range(len(model_grp['nsyns'])),
                'nsyns': model_grp['nsyns'][()]
            })
            nsyns_df = grp_df if nsyns_df is None else pd.concat((nsyns_df, grp_df))

        edges_df = edges_df.merge(nsyns_df, how='left', on=['edge_group_id', 'edge_group_index'])
        edges_df['nsyns'] = edges_df['nsyns'].fillna(1)

        edges_df = edges_df.drop(columns=['edge_group_id', 'edge_group_index'])
        edges_df = edges_df.groupby('edge_type_id').agg(['count', 'sum', 'mean', 'std', 'max', 'min'])
        edges_df.columns = ['_'.join(col).strip() for col in edges_df.columns.values]
        edges_df = edges_df.rename(columns={'nsyns_count': 'n_edges'})
        edges_df = edges_df.merge(edge_types_df, how='left', on='edge_type_id')
        edges_df = edges_df[['edge_type_id', 'source_query', 'target_query', 'n_edges', 'nsyns_sum', 'nsyns_mean', 'nsyns_std', 'nsyns_max', 'nsyns_min']]

        if save_as:
            edges_df.to_csv(save_as, sep=' ', index=False)
        else:
            print(edges_df)


def get_edge_stats(edges_path, edge_types_path, trg_nodes_path, trg_node_types_path, src_nodes_path=None,
                   src_node_types_path=None, group_columns='pop_name', save_as=None):
    if group_columns == 'edge_type_id' or group_columns == ['edge_type_id']:
        get_edge_stats_by_ids(edges_path, edge_types_path)
        return

    group_columns = group_columns if isinstance(group_columns, list) else [group_columns]

    edges_h5 = h5py.File(edges_path, 'r')
    edge_types_df = pd.read_csv(edge_types_path, sep=' ', index_col=False)

    trg_net = sonata.File(data_files=trg_nodes_path, data_type_files=trg_node_types_path)
    src_net = trg_net if src_nodes_path is None else sonata.File(data_files=src_nodes_path, data_type_files=src_node_types_path)

    nsyns_counts_df = None
    for edges_pop, edges_grp in edges_h5['/edges'].items():
        n_edges = len(edges_grp['source_node_id'])
        trg_nodes_pop = edges_grp['target_node_id'].attrs['node_population']
        trg_nodes_pop = trg_nodes_pop if isinstance(trg_nodes_pop, str) else trg_nodes_pop.decode('utf-8')

        src_nodes_pop = edges_grp['source_node_id'].attrs['node_population']
        src_nodes_pop = src_nodes_pop if isinstance(src_nodes_pop, str) else src_nodes_pop.decode('utf-8')

        trg_nodes_df = trg_net.nodes[trg_nodes_pop].to_dataframe(index_by_id=False)
        trg_nodes_df = trg_nodes_df.rename(columns={'node_id': 'target_node_id'})
        # trg_nodes_df.columns = ['target_{}'.format(col) for col in trg_nodes_df.columns.values]
        trg_labels = trg_nodes_df[group_columns[0]]
        for label in group_columns[1:]:
            trg_labels += ',' + trg_nodes_df[label]
        trg_nodes_df['target_label'] = trg_labels
        trg_nodes_df = trg_nodes_df[['target_node_id', 'target_label']]

        src_nodes_df = src_net.nodes[src_nodes_pop].to_dataframe(index_by_id=False)
        src_nodes_df = src_nodes_df.rename(columns={'node_id': 'source_node_id'})
        src_labels = src_nodes_df[group_columns[0]]
        for label in group_columns[1:]:
            src_labels += ',' + src_nodes_df[label]
        src_nodes_df['source_label'] = src_labels
        src_nodes_df = src_nodes_df[['source_node_id', 'source_label']]

        beg_indx = 0
        n_chunks = np.ceil(float(n_edges) / max_load_size).astype(int)
        chunk_num = 1

        print('# Preprocessing edges data.')

        while beg_indx < n_edges:
            end_indx = np.min([beg_indx + max_load_size, n_edges])
            print('#  Chunk {} of {}.'.format(chunk_num, n_chunks))
            chunk_num += 1
            edges_df = pd.DataFrame({
                'source_node_id': edges_grp['source_node_id'][beg_indx:end_indx],
                'target_node_id': edges_grp['target_node_id'][beg_indx:end_indx],
                'edge_group_id': edges_grp['edge_group_id'][beg_indx:end_indx],
                'edge_group_index': edges_grp['edge_group_index'][beg_indx:end_indx],
                'nsyns': 1,
            })

            for group_id in np.unique(edges_grp['edge_group_id'][beg_indx:end_indx]):
                model_grp = edges_grp[str(group_id)]
                if 'nsyns' not in model_grp:
                    continue

                model_indices = edges_df[edges_df['edge_group_id'] == group_id].index.values
                edges_df.loc[model_indices, 'nsyns'] = model_grp['nsyns'][edges_df['edge_group_index']]

            edges_df = edges_df.merge(trg_nodes_df, how='left', on='target_node_id')
            edges_df = edges_df.merge(src_nodes_df, how='left', on='source_node_id')
            edges_df = edges_df[['source_label', 'target_label', 'nsyns']]

            nsyns_counts_df = edges_df if nsyns_counts_df is None else pd.concat([nsyns_counts_df, edges_df])
            beg_indx = end_indx

        print('# Calculating stats.')
        nsyns_stats_df = nsyns_counts_df.groupby(['source_label', 'target_label']).agg(['count', 'sum', 'mean', 'std', 'max'])
        nsyns_stats_df.columns = ['_'.join(col).strip() for col in nsyns_stats_df.columns.values]
        if save_as:
            nsyns_stats_df.to_csv(save_as, sep=' ', index=True)
        else:
            print(nsyns_stats_df)
        print('# Done.')


def cmp_edge_type_counts(edge_type_counts_orig, edge_type_counts_new, save_as=None):
    counts_orig_df = pd.read_csv(edge_type_counts_orig, sep=' ')
    counts_new_df = pd.read_csv(edge_type_counts_new, sep=' ')

    combined_counts_df = counts_orig_df.merge(counts_new_df, how='outer', on=['source_query', 'target_query'], suffixes=['_orig', '_new'])
    combined_counts_df[['counts_orig', 'counts_new']] = combined_counts_df[['counts_orig', 'counts_new']].fillna(0)
    combined_counts_df[['edge_type_id_orig', 'edge_type_id_new']] = combined_counts_df[['edge_type_id_orig', 'edge_type_id_new']].fillna(-1)
    combined_counts_df['diff_raw'] = np.abs(combined_counts_df['counts_orig'] - combined_counts_df['counts_new'])
    combined_counts_df['diff_rel'] = combined_counts_df['diff_raw']/np.max((combined_counts_df['counts_orig'], combined_counts_df['counts_new']))
    combined_counts_df[['counts_orig', 'counts_new', 'edge_type_id_orig', 'edge_type_id_new', 'diff_raw']] = \
        combined_counts_df[['counts_orig', 'counts_new', 'edge_type_id_orig', 'edge_type_id_new', 'diff_raw']].astype(np.int)
    combined_counts_df = combined_counts_df[['counts_orig', 'counts_new', 'diff_raw', 'diff_rel', 'source_query', 'target_query']]

    # combined_counts_df = combined_counts_df.sort_values('counts_orig', ascending=False)
    combined_counts_df = combined_counts_df.sort_values('diff_rel', ascending=False)

    print(combined_counts_df)
    if save_as:
        combined_counts_df.to_csv(save_as, sep=' ', index=False)
    else:
        print(combined_counts_df)


def _check_edge_types_table(orig_edge_types_path, new_edge_types_path):
    orig_edge_types_df = pd.read_csv(orig_edge_types_path, sep=' ', index_col=False, na_values=['NULL', 'NaN'])
    orig_edge_types_df['target_key'] = orig_edge_types_df.apply(
        lambda r: r['target_query'].split('&')[0].split('==')[1].replace("'", ''), axis=1
    )
    orig_edge_types_df['source_key'] = orig_edge_types_df.apply(
        lambda r: r['source_query'].split('&')[0].split('==')[1].replace("'", ''), axis=1
    )

    new_edge_types_df = pd.read_csv(new_edge_types_path, sep=' ', index_col=False)
    new_edge_types_df['target_key'] = new_edge_types_df.apply(
        lambda r: r['target_query'].split('&')[0].split('==')[1].replace("'", ''), axis=1
    )
    new_edge_types_df['source_key'] = new_edge_types_df.apply(
        lambda r: r['source_query'].split('&')[0].split('==')[1].replace("'", ''), axis=1
    )

    merged_edges = orig_edge_types_df.merge(new_edge_types_df, how='outer', on=['target_key', 'source_key'],
                                            suffixes=['_orig', '_new'], indicator=True)
    merged_edges = merged_edges[merged_edges['_merge'] == 'both']

    if 'target_sections_new' in merged_edges and 'target_sections_orig' in merged_edges:
        # NOTE: 'target_sections' may use either " or ', thus comparing the strings
        merged_edges['target_sections_new'] = merged_edges.apply(
            lambda r: r['target_sections_new'].replace('"', "'") if isinstance(r['target_sections_new'], str) else r['target_sections_new'], axis=1
        )
        merged_edges['target_sections_orig'] = merged_edges.apply(
            lambda r: r['target_sections_orig'].replace('"', "'") if isinstance(r['target_sections_orig'], str) else r['target_sections_orig'], axis=1
        )

    for col in ['model_template', 'dynamics_params', 'syn_weight', 'delay', 'distance_range', 'target_sections', 'weight_function', 'weight_sigma']:
        if col not in orig_edge_types_df:
            continue

        col_str = 'checking edge-type column "{}"'.format(col)
        if col not in new_edge_types_df:
            print(' {:>45}: ✗ (not in new edge-types)'.format(col_str))
            continue

        # is both orig and new have null values pandas will not count them as the same
        null_mask = merged_edges['{}_orig'.format(col)].isna() & merged_edges['{}_new'.format(col)].isna()
        tmp_merged_edges = merged_edges[~null_mask]

        if np.issubdtype(tmp_merged_edges['{}_new'.format(col)].dtype, np.number):
            if np.all(np.isnan(tmp_merged_edges['{}_new'.format(col)].values)):
                diff_mask = np.array([True]*len(tmp_merged_edges))
                # print(tmp_merged_edges[['{}_new'.format(col), '{}_orig'.format(col)]].values)
            else:
                diff_mask = ~np.isclose(tmp_merged_edges['{}_new'.format(col)].values, tmp_merged_edges['{}_orig'.format(col)].values, rtol=1e-02, atol=1e-03)
        else:
            diff_mask = tmp_merged_edges['{}_new'.format(col)] != tmp_merged_edges['{}_orig'.format(col)]

        if np.any(diff_mask):
            print('  {:>45}: ✗, invalid (new) edge_type_ids {}'.format(
                col_str,
                tmp_merged_edges[diff_mask]['edge_type_id_new'].values.astype(np.int).tolist()
            ))
        else:
            print('  {:>45}: ✓'.format(col_str))

    merged_edges['edge_type_id_new'] = merged_edges['edge_type_id_new'].astype(np.int)
    merged_edges['edge_type_id_orig'] = merged_edges['edge_type_id_orig'].astype(np.int)
    pivot_table = merged_edges[['edge_type_id_new', 'edge_type_id_orig']]
    pivot_table = pivot_table.rename(columns={'edge_type_id_new': 'edge_type_id'})

    return pivot_table


def _count_edge_types(edges_path, population):
    edges_h5 = h5py.File(edges_path, 'r')
    edges_grp = edges_h5['edges'][population]
    edge_type_ids = edges_grp['edge_type_id'][()]
    edge_type_ids, ids_counts = np.unique(edge_type_ids, return_counts=True)

    return pd.DataFrame({
        'edge_type_id': edge_type_ids,
        'counts': ids_counts
    })


def cmp_edge_types(orig_edges_path, orig_edge_types_path, new_edges_path, new_edge_types_path, population, threshold=0.1):
    title_str = 'comparing {} (new) with {} (orig)'.format(new_edges_path, orig_edges_path)
    print(title_str)
    print('-'*len(title_str))

    # check and compare columns in edge_types_table
    pivot_df = _check_edge_types_table(
        orig_edge_types_path=orig_edge_types_path,
        new_edge_types_path=new_edge_types_path
    )

    # get actual edge counts for each edge_type_id for original network
    orig_edge_counts_df = _count_edge_types(
        edges_path=orig_edges_path,
        population=population
    )

    # get edge counts for new network file
    new_edge_counts_df = _count_edge_types(
        edges_path=new_edges_path,
        population=population
    )
    new_edge_counts_df = new_edge_counts_df.merge(pivot_df, how='left', on='edge_type_id', indicator=True)

    # check if there are edge-types in new network not in original
    unmatch_ids_mask = new_edge_counts_df[new_edge_counts_df['_merge'] != 'both']
    if len(unmatch_ids_mask) > 0:
        print('     new edges_types has an original equivalent: ✗, invalide (new) edge_type_ids {}'.format(
            new_edge_counts_df[unmatch_ids_mask]['edge_type_id'].values
        ))
        new_edge_counts_df = new_edge_counts_df[new_edge_counts_df['_merge'] == 'both']
    else:
        print('     new edges_types has an original equivalent: ✓')

    # The new network may have edge_type_ids that are different from the original
    new_edge_counts_df['actual_edge_type_id'] = new_edge_counts_df['edge_type_id']
    new_edge_counts_df['edge_type_id'] = new_edge_counts_df['edge_type_id_orig']
    new_edge_counts_df = new_edge_counts_df.drop(['edge_type_id_orig', '_merge'], axis=1)

    # combine original and new network edge counts and compare new to original, find absolute + relative differences
    # for each edge-type
    combined_counts_df = orig_edge_counts_df.merge(new_edge_counts_df, how='outer', on='edge_type_id', suffixes=['_orig', '_new'])
    combined_counts_df['counts_orig'] = combined_counts_df['counts_orig'].astype(np.float)
    combined_counts_df['counts_new'] = combined_counts_df['counts_new'].astype(np.float)
    combined_counts_df[['counts_orig', 'counts_new']] = combined_counts_df[['counts_orig', 'counts_new']].fillna(0)
    combined_counts_df['diff_raw'] = np.abs(combined_counts_df['counts_orig'] - combined_counts_df['counts_new'])
    combined_counts_df['diff_rel'] = combined_counts_df['diff_raw'].values / np.max((combined_counts_df['counts_orig'].values, combined_counts_df['counts_new'].values), axis=0)
    combined_counts_df = combined_counts_df.rename(columns={'edge_type_id': 'edge_type_id_orig', 'actual_edge_type_id': 'edge_type_id_new'})
    combined_counts_df = combined_counts_df[['edge_type_id_orig', 'edge_type_id_new', 'counts_orig', 'counts_new', 'diff_raw', 'diff_rel']]
    combined_counts_df = combined_counts_df.sort_values(['diff_rel', 'diff_raw'], ascending=False)

    # mask = combined_counts_df['counts_orig'] > 0.0
    # obs = combined_counts_df[mask]['counts_new'].values
    # exp = combined_counts_df[mask]['counts_orig'].values
    # print(scipy.stats.chisquare(obs, exp))

    rel_thresholds = combined_counts_df['diff_rel'].values > threshold
    if np.any(rel_thresholds):
        print('     relative edge_type counts within threshold: ✗, showing table below (top 10):')
        print(combined_counts_df[rel_thresholds].head())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Build the (biophysical) V1 (and lgn/background inputs) SONATA network files')
    parser.add_argument('--orig-network-dir', default='network_orig')

    parser.add_argument('--v1', action='store_true', default=False)
    parser.add_argument('--lgn', action='store_true', default=False)
    parser.add_argument('--bkg', action='store_true', default=False)
    parser.add_argument('--validate', action='store_true', default=False)
    parser.add_argument('--cmp', action='store_true', default=False)
    parser.add_argument('--stats', action='store_true', default=False)
    parser.add_argument('--exclude-nodes', action='store_true', default=False)
    parser.add_argument('--exclude-edges', action='store_true', default=False)
    parser.add_argument('network_dirs', type=str, nargs='+')
    args = parser.parse_args()

    all_nets = not (args.v1 or args.lgn or args.bkg)
    all_checks = not (args.validate or args.cmp or args.stats)
    for network_dir in args.network_dirs:
        for net in ['v1', 'lgn', 'bkg']:
            if not (all_nets or getattr(args, net)):
                continue

            new_nodes_path = os.path.join(network_dir, '{}_nodes.h5'.format(net))
            new_node_types_path = os.path.join(network_dir, '{}_node_types.csv'.format(net))
            new_edges_path = os.path.join(network_dir, '{}_v1_edges.h5'.format(net))
            new_edge_types_path = os.path.join(network_dir, '{}_v1_edge_types.csv'.format(net))

            if all_checks or args.validate:
                if not args.exclude_nodes:
                    validate_nodes(
                        nodes_path=new_nodes_path,
                        node_types_path=new_node_types_path,
                        population=net
                    )
                    print()

                if not args.exclude_edges:
                    validate_edges(
                        edges_path=new_edges_path,
                        edge_types_path=new_edge_types_path,
                        trg_nodes_path=os.path.join(network_dir, 'v1_nodes.h5'),
                        src_nodes_path=new_nodes_path,
                        # models_dir=None
                    )
                    print()

            if all_checks or args.cmp:
                orig_nodes_path = os.path.join(args.orig_network_dir, '{}_nodes.h5'.format(net))
                orig_node_types_path = os.path.join(args.orig_network_dir, '{}_node_types.csv'.format(net))
                orig_edges_path = os.path.join(args.orig_network_dir, '{}_v1_edges.h5'.format(net))
                orig_edge_types_path = os.path.join(args.orig_network_dir, '{}_v1_edge_types.csv'.format(net))

                if not args.exclude_nodes:
                    cmp_node_types(
                        orig_nodes_path=orig_nodes_path,
                        orig_node_types_path=orig_node_types_path,
                        new_nodes_path=new_nodes_path,
                        new_node_types_path=new_node_types_path,
                        population=net,
                        grp_cols=['dynamics_params', 'morphology', 'pop_name', 'model_type', 'model_template'] if net == 'v1' else ['pop_name'],
                        check_coords=False
                    )
                    print()

                if not args.exclude_edges:
                    cmp_edge_types(
                        orig_edges_path=orig_edges_path,
                        orig_edge_types_path=orig_edge_types_path,
                        new_edges_path=new_edges_path,
                        new_edge_types_path=new_edge_types_path,
                        population='{}_to_v1'.format(net)
                    )
                    print()

            if args.stats:
                if not args.exclude_nodes:
                    get_node_stats(
                        nodes_path=new_nodes_path,
                        node_types_path=new_node_types_path,
                        population=net,
                        group_columns=['node_type_id', 'pop_name']
                    )
                    print()

                if not args.exclude_edges:
                    get_edge_stats(
                        edges_path=new_edges_path,
                        edge_types_path=new_edge_types_path,
                        trg_nodes_path=os.path.join(network_dir, 'v1_nodes.h5'),
                        trg_node_types_path=os.path.join(network_dir, 'v1_node_types.csv'),
                        src_nodes_path=new_nodes_path,
                        src_node_types_path=new_node_types_path,
                        group_columns='edge_type_id'
                        # group_columns='pop_name'
                    )

