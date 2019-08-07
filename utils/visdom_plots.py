import sys
import visdom
import numpy as np
from subprocess import Popen, PIPE


def create_visdom_connections(port):
    """If the program could not connect to Visdom server, this function will start a new server at port < port > """
    cmd = sys.executable + ' -m visdom.server -p %d &>/dev/null &' % port
    print('\n\nCould not connect to Visdom server. \n Trying to start a server....')
    print('Command: %s' % cmd)
    Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE)


class VisdomLogger(dict):
    """Plot losses."""

    def __init__(self, port, *args, **kwargs):
        self.visdom_port = port
        self.visdom = visdom.Visdom(port=port)
        if not self.visdom.check_connection():
            create_visdom_connections(port)
        super(VisdomLogger, self).__init__(*args, **kwargs)
        self.registered = False
        self.plot_attributes = {}

    def register_keys(self, keys):
        for key in keys:
            self[key] = []
        for i, key in enumerate(self.keys()):
            self.plot_attributes[key] = {'win_id': i}
        self.registered = True

    def update(self, new_records):
        """Add new updates to records.

        :param new_records: dict of new updates. example: {'loss': [10., 5., 2.], 'lr': [1e-3, 1e-3, 5e-4]}
        """
        for key, val in new_records.items():
            if key in self.keys():
                self[key].extend(val)

    def do_plotting(self):
        for k in self.keys():
            y_values = np.array(self[k])
            x_values = np.arange(len(self[k]))
            self.visdom.line(Y=y_values, X=x_values, win=self.plot_attributes[k]['win_id'],
                             opts={'title': k.upper()}, update='append')
            # self[k] = []
