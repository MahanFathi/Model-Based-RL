import torch
import torch.nn as nn
from model.layers.feed_forward import FeedForward


class Basic(nn.Module):
    """This architecture takes in the meta data, turns it into the
    initial hidden state of the LSTM, and flows it w/ the sequential data.
    """
    def __init__(self, cfg):
        """Build the model from the fed config node.

        :param cfg: CfgNode containing the configurations of everything.
        """
        super(Basic, self).__init__()
        self.cfg = cfg
        # build meta data to initial hidden
        metadata_fc_output_size = cfg.MODEL.RNN.HIDDEN_SIZE * cfg.MODEL.RNN.NUM_LAYERS
        if cfg.MODEL.METADATA.FC.INITIALIZE_CELL_STATE:
            metadata_fc_output_size *= 2
        self.metadata_to_initial_hiddens = FeedForward(cfg.INPUT.METADATA.FEATURE_SIZE, cfg.MODEL.METADATA.FC.LAYERS,
                                                       metadata_fc_output_size, cfg.MODEL.METADATA.FC.NORM_LAYERS)

        # build sequential data to input lstm
        self.sequential_to_embedding = FeedForward(cfg.INPUT.SEQUENTIAL_DATA.FEATURE_SIZE, cfg.MODEL.RNN.FC.INPUT.LAYERS,
                                                   cfg.MODEL.RNN.FC.INPUT.OUT_SIZE, cfg.MODEL.RNN.FC.INPUT.NORM_LAYERS)

        # build main LSTM
        lstm_input_size = cfg.MODEL.RNN.FC.INPUT.OUT_SIZE
        self.lstm = nn.LSTM(lstm_input_size, cfg.MODEL.RNN.HIDDEN_SIZE, cfg.MODEL.RNN.NUM_LAYERS,
                            batch_first=True, dropout=cfg.MODEL.RNN.DROPOUT)

        # build LSTM final output fully connected layer
        self.lstm_output_fc = FeedForward(cfg.MODEL.RNN.HIDDEN_SIZE, cfg.MODEL.RNN.FC.OUTPUT.LAYERS,
                                          cfg.MODEL.RNN.FC.OUTPUT.OUT_SIZE, cfg.MODEL.RNN.FC.OUTPUT.NORM_LAYERS,
                                          relu_out=False)

        # build loss criterion
        self.loss_criterion = nn.MSELoss()

    def _forward(self, meta_data, seq_data, seq_lengths):
        """Flow the graph.
        NOTE: As we use mini-batches of sequential data, `seq_data` should be padded when getting items from data-loader.
        :param meta_data: torch.tensor of size (batch_size, cfg.INPUT.META_DATA.FEATURE_SIZE)
        :param seq_data: torch.tensor of size (batch_size, max_sequence_size, cfg.INPUT.SEQUENTIAL_DATA.FEATURE_SIZE)
        :param seq_lengths: torch.tensor of size (batch_size, )
        :return: out: torch.tensor of size (batch_size, cfg.MODEL.RNN.FC.OUTPUT.OUT_SIZE)
        """

        batch_size = meta_data.size()[0]
        sequence_size = seq_data.size()[1]

        # forward to metadata fc graph to get the initial hidden/cell states
        lstm_initials_flat = self.metadata_to_initial_hiddens(meta_data)
        initial_hidden_state, initial_cell_state = self._get_initial_hidden_states(lstm_initials_flat)

        # construct inputs from sequential_to_embedding
        seq_batched_seq_data = seq_data.view([-1, self.cfg.INPUT.SEQUENTIAL_DATA.FEATURE_SIZE])     # reshape for fc
        seq_batched_embeddings = self.sequential_to_embedding(seq_batched_seq_data)  # get embeddings of all sequences
        embedding_sequences = seq_batched_embeddings.view([batch_size, sequence_size,
                                                           self.cfg.MODEL.RNN.FC.INPUT.OUT_SIZE])

        # hide padded data from LSTM via pack_padded_sequence and pad_packed_sequence
        # note that seq_data should be in descending order of sequence length
        embedding_sequences = torch.nn.utils.rnn.pack_padded_sequence(embedding_sequences, seq_lengths, batch_first=True)
        # feed to LSTM, outputs a packed sequence
        lstm_output, _ = self.lstm(embedding_sequences, (initial_hidden_state, initial_cell_state))
        # undo the packing
        lstm_output, _ = torch.nn.utils.rnn.pad_packed_sequence(lstm_output, batch_first=True)
        # pick the last hidden neurons of each sequence and run it trough the fully connected output network
        last_hiddens = lstm_output[torch.arange(batch_size).tolist(), [seq_len - 1 for seq_len in seq_lengths.tolist()], :]
        out = self.lstm_output_fc(last_hiddens)
        return out  # the padded output should be taken into account by the loss function

    def _get_initial_hidden_states(self, lstm_initials_flat):
        if self.cfg.MODEL.METADATA.FC.INITIALIZE_CELL_STATE:
            # note that initial hidden and cell state shapes don't adhere to `batch_first`
            lstm_initials = lstm_initials_flat.view([self.cfg.MODEL.RNN.NUM_LAYERS, -1,
                                                     self.cfg.MODEL.RNN.HIDDEN_SIZE * 2])
            initial_hidden_state, initial_cell_state = torch.split(lstm_initials, self.cfg.MODEL.RNN.HIDDEN_SIZE, 2)
        else:
            lstm_initials = lstm_initials_flat.view([self.cfg.MODEL.RNN.NUM_LAYERS, -1,
                                                     self.cfg.MODEL.RNN.HIDDEN_SIZE])
            initial_hidden_state = lstm_initials
            initial_cell_state = torch.zeros_like(initial_hidden_state).to(self.cfg.MODEL.DEVICE)
        return initial_hidden_state, initial_cell_state

    def forward(self, meta_data, seq_data, seq_lengths, eta=None):
        """

        :param meta_data: torch.tensor of size (batch_size, cfg.INPUT.META_DATA.FEATURE_SIZE)
        :param seq_data: torch.tensor of size (batch_size, max_sequence_size, cfg.INPUT.SEQUENTIAL_DATA.FEATURE_SIZE)
        :param seq_lengths: torch.tensor of size (batch_size, )
        :param eta: torch.tensor of size (batch_size, 1)
        :return: TRAINING MODE: LOSS float, INFERENCE_MODE: PREDICTION torch.tensor of size (batch_size, 1)
        """
        if self.training and eta is None:
            raise ValueError("In training mode, eta targets should be passed.")

        out = self._forward(meta_data, seq_data, seq_lengths)

        if self.training:
            return self.loss_criterion(out, eta)

        return out


