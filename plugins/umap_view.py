import numpy as np
from phy import IPlugin, Bunch
#from phy.apps.base import get_best_channel as best 
from phy.cluster.views import ScatterView


#def umap(x):
#    """Perform the dimension reduction of the array x."""
#    from umap import UMAP
#    return UMAP().fit_transform(x)


class WaveformUMAPView(ScatterView):
    """Every view corresponds to a unique view class, so we need to subclass ScatterView."""
    pass


class umap_view(IPlugin):
    def attach_to_controller(self, controller):
        def coords(cluster_ids):
            spike_ids = controller.selector.select_spikes(cluster_ids)
	        # We get the cluster ids corresponding to the chosen spikes.
            spike_clusters = controller.supervisor.clustering.spike_clusters[spike_ids]
            #ch0 = controller.get_best_channels(cluster_ids)
            #ch1 = (ch0-1)
            #ch1 = controller.get_best_channel(cluster_ids)
            #ch2 = (cluster_site1 - 1)
            #a = controller.get_best_channel(spike_ids)
            #b = controller.get_best_channels(2)[1])

            data1 = controller.get_spike_raw_amplitudes(spike_ids, channel_id=controller.get_best_channels(spike_clusters[0])[0])
            data2 = controller.get_spike_raw_amplitudes(spike_ids, channel_id=controller.get_best_channels(spike_clusters[0])[1])
            print(controller.get_best_channels(spike_clusters[0])[0], controller.get_best_channels(spike_clusters[0])[1])
            #data1 = controller.get_spike_raw_amplitudes(spike_ids, channel_id=controller.get_best_channels)
            #data2 = controller.get_spike_raw_amplitudes(spike_ids, channel_id=controller.get_best_channels)

            #data2 = controller.get_spike_raw_amplitudes(spike_ids, channel_id=controller.get_best_channels(1)[0])
            #data2 = controller.model.get_spike_raw_amplitudes(spike_ids)
            #(n_spikes, n_samples, n_channels) = data1.shape
            #(n_spikes, n_samples, n_channels) = data2.shape
            #data1out = data1.transpose((0, 2, 1))  # get an (n_spikes, n_channels, n_samples) array
            #data2out = data2.reshape((n_spikes, n_samples * n_channels))
            #pos = np.array([[1,2],[2,3],[4,5]])
            #pos = np.array([data1], [data2])
            pos = np.ones((len(data1), 2))
            pos[:,0] = data1
            pos[:,1] = data2

            #pos = (data1, data2)
            return Bunch(pos=pos, spike_ids=spike_ids, spike_clusters=spike_clusters)

        def create_view():
            """Create and return a histogram view."""
            return WaveformUMAPView(coords=controller.context.cache(coords))

        # Maps a view name to a function that returns a view
        # when called with no argument.
        controller.view_creator['WaveformUMAPView'] = create_view
