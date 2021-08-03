import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# from bmtk.analyzer.spike_trains import plot_raster
from bmtk.utils import sonata


def compare_raster(spikes_paths, network_dirs, max_time=1000.0, titles=None, col_names=None, save_as=None):
    assert(len(spikes_paths) == len(network_dirs))  # check equal num of columns
    assert(all([len(l) == len(spikes_paths[0]) for l in spikes_paths]))  # check there is an equal num of rows

    n_cols = len(spikes_paths)
    n_rows = len(spikes_paths[0])

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(11 + n_cols*2, 3 + n_rows*2))
    axes = np.array([axes]).reshape(1, 1) if n_cols == n_rows == 1 else axes.reshape(n_rows, n_cols)
    color_map = {}
    for c in range(n_cols):
        net = sonata.File(
            data_files=os.path.join(network_dirs[c], 'v1_nodes.h5'),
            data_type_files=os.path.join(network_dirs[c], 'v1_node_types.csv')
        )
        nodes_df = net.nodes['v1'].to_dataframe(index_by_id=False)
        nodes_df = nodes_df[['node_id', 'pop_name', 'x', 'y', 'z']]
        nodes_df = nodes_df.rename({'node_id': 'node_ids'}, axis=1)
        nodes_df = nodes_df.sort_values(['pop_name', 'x', 'y', 'z'])
        nodes_df['ordered_ids'] = np.arange(0, len(nodes_df))

        for r in range(n_rows):
            spikes_df = pd.read_csv(spikes_paths[c][r], sep=' ', usecols=['node_ids', 'timestamps'])
            spikes_df = spikes_df[spikes_df['timestamps'] <= max_time]

            spikes_nodes_df = spikes_df.merge(nodes_df, how='left', on='node_ids')
            spikes_nodes_df = spikes_nodes_df.sort_values('pop_name')

            ax = axes[r, c]
            for pop_name, subset_df in spikes_nodes_df.groupby('pop_name'):
                # ax.scatter(subset_df['timestamps'], subset_df['ordered_ids'], lw=0, s=1, label=pop_name)
                if pop_name in color_map:
                    ax.scatter(subset_df['timestamps'], subset_df['ordered_ids'], lw=0, s=1, label=pop_name, c=color_map[pop_name])
                else:
                    p = ax.scatter(subset_df['timestamps'], subset_df['ordered_ids'], lw=0, s=1, label=pop_name)
                    color_map[pop_name] = p.get_facecolor()

            ax.set_xlim((0, max_time))
            if r < n_rows - 1:
                ax.set_xticklabels([])

            if r == 0:
                ax.legend(fontsize=5, loc='upper right')

            if titles is not None and c == 0:
                ax.set_ylabel(titles[r])

            if col_names is not None and r == (n_rows-1):
                ax.set_xlabel(col_names[c])

    plt.tight_layout()

    if save_as:
        plt.savefig(save_as)
    plt.show()


if __name__ == '__main__':
    compare_raster(
        network_dirs=[
            'network_rebuilt',
            'network_orig'
        ],
        spikes_paths=[
            ['output_norecurrent_nobkg_rebuilt/spikes.csv', 'output_norecurrent_nolgn_rebuilt/spikes.csv', 'output_rebuilt/spikes.csv'],
            ['output_norecurrent_nobkg_orig/spikes.csv', 'output_norecurrent_nolgn_orig/spikes.csv', 'output_orig/spikes.csv']
        ],
        titles=['lgn only', 'bkg only', 'full'],
        col_names=['rebuilt', 'original'],
        # save_as='v1_bionet.orig_v_new.png'
    )

