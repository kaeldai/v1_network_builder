import os
import sys
import json
import numpy as np
import pandas as pd
import argparse
import logging

from node_funcs import generate_random_positions, generate_positions_grids, get_filter_spatial_size, get_filter_temporal_params
from edge_funcs import compute_pair_type_parameters, connect_cells, select_lgn_sources, selected_src_types, src_node_ids
from bmtk.builder import NetworkBuilder


logger = logging.getLogger(__name__)


def add_nodes_v1(fraction=.50, node_props='biophys_props/v1_node_models.json'):
    v1_models = json.load(open(node_props, 'r'))

    inner_radial_range = v1_models['inner_radial_range']
    outer_radial_range = v1_models['outer_radial_range']

    net = NetworkBuilder('v1')

    for location, loc_dict in v1_models['locations'].items():
        for pop_name, pop_dict in loc_dict.items():
            pop_size = pop_dict['ncells']
            depth_range = -np.array(pop_dict['depth_range'], dtype=np.float)
            ei = pop_dict['ei']

            for model in pop_dict['models']:
                if 'N' not in model:
                    # Assumes a 'proportion' key with a value from 0.0 to 1.0, N will be a proportion of pop_size
                    model['N'] = model['proportion']*pop_size
                    del model['proportion']

                if fraction != 1.0:
                    # Each model will use only a fraction of the of the number of cells for each model
                    # NOTE: We are using a ceiling function so there is atleast 1 cell of each type - however for models
                    #  with only a few initial cells they can be over-represented.
                    model['N'] = int(np.ceil(fraction*model['N']))

                # create a list of randomized cell positions for each cell type
                radial_range = inner_radial_range if model['model_type'] == 'biophysical' else outer_radial_range
                N = model['N']
                positions = generate_random_positions(N, depth_range, radial_range)

                # properties used to build the cells for each cell-type
                node_props = {
                    'N': N,
                    'node_type_id': model['node_type_id'],
                    'model_type': model['model_type'],
                    'model_template': model['model_template'],
                    'dynamics_params': model['dynamics_params'],
                    'ei': ei,
                    'location': location,
                    'pop_name': ('LIF' if model['model_type'] == 'point_process' else '') + pop_name,
                    'population': 'v1',
                    'x': positions[:, 0],
                    'y': positions[:, 1],
                    'z': positions[:, 2],
                    'tuning_angle': np.linspace(0.0, 360.0, N, endpoint=False),
                }
                if model['model_type'] == 'biophysical':
                    # for biophysically detailed cell-types add info about rotations and morphollogy
                    node_props.update({
                        'rotation_angle_xaxis': np.zeros(N),  # for RTNeuron store explicity store the x-rotations (even though it should be 0 by default).
                        'rotation_angle_yaxis': 2*np.pi*np.random.random(N),
                        'rotation_angle_zaxis': np.full(N, model['rotation_angle_zaxis']),  # for RTNeuron we need to store z-rotation in the h5 file.
                        'model_processing': model['model_processing'],
                        'morphology': model['morphology']
                    })

                net.add_nodes(**node_props)

    return net


def find_direction_rule(src_label, trg_label):
    src_ei = 'e' if src_label.startswith('e') or src_label.startswith('LIFe') else 'i'
    trg_ei = 'e' if trg_label.startswith('e') or trg_label.startswith('LIFe') else 'i'

    if src_ei == 'e' and trg_ei == 'e':
        return 'DirectionRule_EE', 30.0

    elif src_ei == 'e' and trg_ei == 'i':
        return 'DirectionRule_others', 90.0

    elif src_ei =='i' and trg_ei == 'e':
        return 'DirectionRule_others', 90.0

    else:
        return 'DirectionRule_others', 50.0


def add_edges_v1(net):
    conn_weight_df = pd.read_csv('biophys_props/v1_edge_models.csv', sep=' ')

    conn_weight_df = conn_weight_df[~(conn_weight_df['source_label'] == 'LGN')]
    for _, row in conn_weight_df.iterrows():
        node_type_id = row['target_model_id']
        src_type = row['source_label']
        trg_type = row['target_label']
        src_trg_params = compute_pair_type_parameters(src_type, trg_type)

        weight_fnc, weight_sigma = find_direction_rule(src_type, trg_type)
        if src_trg_params['A_new'] > 0.0:
            if trg_type.startswith('LIF'):
                net.add_edges(
                    source={'pop_name': src_type},
                    target={'node_type_id': node_type_id},
                    iterator='all_to_one',
                    connection_rule=connect_cells,
                    connection_params={'params': src_trg_params},
                    dynamics_params=row['params_file'],
                    model_template='exp2syn',
                    syn_weight=row['weight_max'],
                    delay=row['delay'],
                    weight_function=weight_fnc,
                    weight_sigma=weight_sigma
                )
            else:
                net.add_edges(
                    source={'pop_name': src_type},
                    target={'node_type_id': node_type_id},
                    iterator='all_to_one',
                    connection_rule=connect_cells,
                    connection_params={'params': src_trg_params},
                    dynamics_params=row['params_file'],
                    model_template='exp2syn',
                    syn_weight=row['weight_max'],
                    delay=row['delay'],
                    weight_function=weight_fnc,
                    weight_sigma=weight_sigma,
                    distance_range=row['distance_range'],
                    target_sections=row['target_sections']
                )
    return net


def add_nodes_lgn():
    lgn_models = json.load(open('node_props/lgn_models.json', 'r'))

    lgn = NetworkBuilder('lgn')
    X_grids = 15  # 15#15      #15
    Y_grids = 10  # 10#10#10      #10
    X_len = 240.0  # In linear degrees
    Y_len = 120.0  # In linear degrees

    xcoords = []
    ycoords = []
    for model, params in lgn_models.items():
        # Get position of lgn cells and keep track of the averaged location
        # For now, use randomly generated values
        total_N = params['N'] * X_grids * Y_grids

        # Get positional coordinates of cells
        positions = generate_positions_grids(params['N'], X_grids, Y_grids, X_len, Y_len)
        xcoords += [p[0] for p in positions]
        ycoords += [p[1] for p in positions]

        # Get spatial filter size of cells
        filter_sizes = get_filter_spatial_size(params['N'], X_grids, Y_grids, params['size_range'])

        # Get filter temporal parameters
        filter_params = get_filter_temporal_params(params['N'], X_grids, Y_grids, model)

        # Get tuning angle for LGN cells
        # tuning_angles = get_tuning_angles(params['N'], X_grids, Y_grids, model)

        lgn.add_nodes(
            N=total_N,
            pop_name=params['model_id'],
            model_type='virtual',
            ei='e',
            location='LGN',
            x=positions[:, 0],
            y=positions[:, 1],
            spatial_size=filter_sizes,
            kpeaks_dom_0=filter_params[:, 0],
            kpeaks_dom_1=filter_params[:, 1],
            weight_dom_0=filter_params[:, 2],
            weight_dom_1=filter_params[:, 3],
            delay_dom_0=filter_params[:, 4],
            delay_dom_1=filter_params[:, 5],
            kpeaks_non_dom_0=filter_params[:, 6],
            kpeaks_non_dom_1=filter_params[:, 7],
            weight_non_dom_0=filter_params[:, 8],
            weight_non_dom_1=filter_params[:, 9],
            delay_non_dom_0=filter_params[:, 10],
            delay_non_dom_1=filter_params[:, 11],
            tuning_angle=filter_params[:, 12],
            sf_sep=filter_params[:, 13],
        )

    return lgn


def add_lgn_v1_edges(v1_net, lgn_net, x_len=240.0, y_len=120.0):
    conn_weight_df = pd.read_csv('conn_props/edge_type_models.csv', sep=' ')
    conn_weight_df = conn_weight_df[(conn_weight_df['source_label'] == 'LGN')]
    # conn_weight_df = conn_weight_df[conn_weight_df['target_model_id'] == 100000101]

    # print(conn_weight_df)
    # exit()


    lgn_mean = (x_len/2.0, y_len/2.0)
    lgn_models = json.load(open('node_props/lgn_models.json', 'r'))

    i = 0

    for _, row in conn_weight_df.iterrows():
        # i += 1
        # if i > 1:
        #     break

        src_type = row['source_label']
        trg_type = row['target_label']
        target_node_type = row['target_model_id']

        # print(src_type, trg_type)
        #
        # target_nodes = list(v1_net.nodes(node_type_id=target_node_type))
        # for node in target_nodes:
        #     print(node.node_id, node['model_type'])
        #     assert(node['model_type'] == 'point_process')
        # exit()
        # print(target_node_type, row['weight_max'])


        edge_params = {
            'source': lgn_net.nodes(location='LGN'),
            'target': v1_net.nodes(node_type_id=target_node_type),
            'iterator': 'all_to_one',
            'connection_rule': select_lgn_sources,
            'connection_params': {'lgn_mean': lgn_mean, 'lgn_models': lgn_models},
            'dynamics_params': row['params_file'],
            'model_template': None if trg_type.startswith('LIF') else 'exp2syn',
            'syn_weight': row['weight_max'],
            'delay': row['delay'],
            'weight_function': row['weight_func'],
            'weight_sigma': row['weight_sigma']
        }
        if row['target_sections'] is not None:
            edge_params.update({
                'target_sections': row['target_sections'],
                'distance_range': row['distance_range']
            })

        lgn_net.add_edges(**edge_params)

    # exit()
    return lgn_net


def add_nodes_bkg():
    bkg = NetworkBuilder('bkg')
    bkg.add_nodes(
        N=1, pop_name='SG_001', ei='e', location='BKG',
        model_type='virtual',
        x=[-91.23767151810344],
        y=[233.43548226294524]
    )
    return bkg


def add_bkg_v1_edges(v1_net, bkg_net):
    conn_weight_df = pd.read_csv('conn_props/bkg_edge_type_models.csv', sep=' ')

    for _, row in conn_weight_df.iterrows():
        # src_type = row['source_label']
        trg_type = row['target_label']
        target_node_type = row['target_model_id']
        nsyns = row.get('nsyns', 1)

        edge_params = {
            'source': bkg_net.nodes(),
            'target': v1_net.nodes(node_type_id=target_node_type),
            'connection_rule': lambda s, t, n: n,
            'connection_params': {'n': nsyns},
            'dynamics_params': row['dynamics_params'],
            'syn_weight': row['syn_weight'],
            'delay': row['delay'],
        }
        if trg_type == 'biophysical':
            edge_params.update({
                'model_template': 'exp2syn',
                'target_sections': row['target_sections'],
                'distance_range': row['distance_range']
            })
        bkg_net.add_edges(**edge_params)

    return bkg_net


def check_files_exists(output_dir, src_net, trg_net, force_overwrite):
    if force_overwrite:
        return

    files = [os.path.join(output_dir, '{}_nodes.h5'.format(src_net)),
             os.path.join(output_dir, '{}_node_types.csv'.format(src_net)),
             os.path.join(output_dir, '{}_{}_edges.h5'.format(src_net, trg_net)),
             os.path.join(output_dir, '{}_{}_edge_types.csv'.format(src_net, trg_net))]
    for f in files:
        if os.path.exists(f):
            raise Exception('file {} already exists. Use --force-overwrite to overwrite exists file or --output-dir to change path to file'.format(f))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Build the (biophysical) V1 (and lgn/background inputs) SONATA network files')
    parser.add_argument('-o', '--output-dir', default='network', help='directory where network files will be saved.')
    parser.add_argument('--v1-nodes-dir',
                        help='directory containing existing v1 nodes. Used when building lgn/bkg network only')
    parser.add_argument('-f', '--force-overwrite', action='store_true', default=False,
                        help='force existings network files to be overwritten')
    parser.add_argument('--fraction', type=float, default=1.0,
                        help='Specify a value between (0, 1.0) to build a network with only a given fraction of the V1 nodes')
    parser.add_argument('--debug', action='store_true', default=False, help='logs debugging info')
    parser.add_argument('--log-file', type=str, default=None, help='log build process to a file.')
    parser.add_argument('networks', type=str, nargs='*', default=['v1', 'bkg', 'lgn'])
    args = parser.parse_args()

    np.random.seed(100)

    nets = set(args.networks)
    if nets - {'v1', 'lgn', 'bkg'}:
        # check specified networks
        raise Exception('Uknown network(s) {}. valid networks: v1, lgn, bkg'.format(set(nets) - {'v1', 'lgn', 'bkg'}))

    # if not os.path.exists(args.output_dir):
    #     os.mkdir(args.output_dir)

    # set up logging
    logging.basicConfig(
        format='%(asctime)s [%(levelname)s] %(message)s',
        level=logging.DEBUG if args.debug else logging.INFO,
        filename=args.log_file
    )

    logger.info('$ ' + ' '.join(sys.argv))
    v1 = None
    if 'v1' in nets:
        logger.info('Building v1 network')
        check_files_exists(args.output_dir, 'v1', 'v1', args.force_overwrite)
        v1 = add_nodes_v1(fraction=args.fraction)
        v1 = add_edges_v1(v1)
        v1.build()
        v1.save(args.output_dir)
        nets.remove('v1')

    if len(nets) == 0:
        exit(0)

    if v1 is None:
        logger.info('Loading in v1 nodes from {} ...'.format(args.v1_nodes_dir))
        v1 = NetworkBuilder('v1')
        v1.import_nodes(os.path.join(args.v1_nodes_dir, 'v1_nodes.h5'),
                        os.path.join(args.v1_nodes_dir, 'v1_node_types.csv'))
        print('...  done.')

    if 'lgn' in nets:
        logger.info('Building lgn network.')
        check_files_exists(args.output_dir, 'lgn', 'v1', args.force_overwrite)
        lgn = add_nodes_lgn()
        lgn = add_lgn_v1_edges(v1, lgn)
        lgn.build()
        lgn.save(args.output_dir)

        print(selected_src_types)
        # print(list(src_node_ids))

    if 'bkg' in nets:
        logger.info('Building bkg network.')
        check_files_exists(args.output_dir, 'bkg', 'v1', args.force_overwrite)
        bkg = add_nodes_bkg()
        bkg = add_bkg_v1_edges(v1, bkg)
        bkg.build()
        bkg.save(args.output_dir)

    logger.info('$ done.')
