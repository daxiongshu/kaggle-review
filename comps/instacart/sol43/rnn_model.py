"""
this is based on 43th solution
https://github.com/colinmorris/instacart-basket-prediction
https://www.kaggle.com/c/instacart-market-basket-analysis/discussion/38159
"""
from models.tf_models.BaseRnnModel import BaseRnnModel
from comps.instacart.sol43.constant import N_PRODUCTS, N_AISLES, N_DEPARTMENTS
class rnnModel(BaseRnnModel):

    def __init__(self,flags,data):
        super().__init__(flags,data)
        self.data = data

    
    def _build(self, input_vars):
        flags = self.flags
        self.cell = self._build_cell()
        # TODO: don't really need to attach all these inputs to self anymore now that
        # we're not using placeholders. But just lazily minimizing code changes.
        input_data = input_vars['features']
        max_seq_len = tf.shape(self.input_data)[1]
        batch_size = tf.shape(self.input_data)[0]
        label_shape = [self.batch_size, self.max_seq_len]
        # TODO: Some of these vars aren't needed depending on mode
        sequence_lengths = input_vars['seqlen']

        cell_input = self.input_data

        embedding_inputs = self._build_embedding_inputs(input_vars)
        if embedding_inputs:
            cell_input = tf.concat([self.input_data]+embedding_inputs, 2)

        initial_state = self.cell.zero_state(batch_size=self.batch_size,
            dtype=tf.float32)
        output, last_state = tf.nn.dynamic_rnn(
            self.cell, 
            cell_input,
            sequence_length=self.sequence_lengths,
            # this kwarg is optional, but docs aren't really clear on what
            # happens if it isn't provided. Probably just the zero state,
            # so this isn't necessary. But whatever.
            # yeah, source says zeros. But TODO should prooobably be documented?
            initial_state=initial_state,
            dtype=tf.float32,
        )
    # TODO: would like to log forgettitude, but this seems quite tricky. :(
    with tf.variable_scope('RNN'):
      output_w = tf.get_variable('output_w', [self.hps.rnn_size, 1])
      output_b = tf.get_variable('output_b', [1])
    
    output = tf.reshape(output, [-1, self.hps.rnn_size])
    logits = tf.nn.xw_plus_b(output, output_w, output_b)
    logits = tf.reshape(logits, label_shape)
    self.logits = logits
    # The logits that were actually relevant to prediction/loss
    boolmask = tf.cast(self.lossmask, tf.bool)
    used_logits = tf.boolean_mask(self.logits, boolmask)
    self.add_summary( 'Logits', used_logits )
    loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.labels, logits=logits)
    # apply loss mask
    loss = tf.multiply(loss, self.lossmask)
    # Total loss per sequence
    loss_per_seq = tf.reduce_sum(loss, axis=1)
    # mean loss per sequence (averaged over number of sequence elements not 
    # zeroed out by lossmask - no free rides)
    # tf has way too many ways to do division :[
    loss_per_seq = tf.realdiv(loss_per_seq, 
        tf.reduce_sum(self.lossmask, axis=1)
    )
    weighted_loss_per_seq = tf.multiply(loss_per_seq, input_vars['weight'])
    # Loss on just the last element of each sequence.
    last_order_indices = self.sequence_lengths - 1 
    r = tf.range(self.batch_size)
    finetune_indices = tf.stack([r, last_order_indices], axis=1)
    self.finetune_cost = tf.reduce_mean(
        tf.gather_nd(loss, finetune_indices)
    )
    self.lastorder_logits = tf.gather_nd(logits, finetune_indices)
    
    self.cost = tf.reduce_mean(loss_per_seq)
    self.weighted_cost = tf.reduce_mean(weighted_loss_per_seq)
    self.total_cost = self.cost
    if self.hps.l2_weight:
      tvs = tf.trainable_variables()
      # Penalize everything except for biases
      l2able_vars = [v for v in tvs if ('bias' not in v.name and 'output_b' not in v.name)]
      self.weight_penalty = tf.add_n([
          tf.nn.l2_loss(v) 
          for v in l2able_vars]) * self.hps.l2_weight
      self.total_cost = tf.add(self.cost, self.weight_penalty)
    else:
      self.weight_penalty = tf.constant(0)

    if self.hps.is_training:
        self.lr = tf.Variable(self.hps.learning_rate, trainable=False)
        if self.hps.optimizer == 'Adam':
          optimizer_fn = tf.train.AdamOptimizer
        elif self.hps.optimizer == 'LazyAdam':
          optimizer_fn = tf.contrib.opt.LazyAdamOptimizer
        else:
          assert False, "Don't know about {} optimizer".format(self.hps.optimizer)
        self.optimizer = optimizer_fn(self.lr)
        optimizer = self.optimizer
        if self.hps.grad_clip:
          gvs = optimizer.compute_gradients(self.total_cost)
          g = self.hps.grad_clip
          capped_gvs = [ (tf.clip_by_value(grad, -g, g), var) for grad, var in gvs]
          self.train_op = optimizer.apply_gradients(
              capped_gvs, global_step=self.global_step
          )
        else:
          self.train_op = optimizer.minimize(
                  self.total_cost,
                  self.global_step,
          )
        if self.hps.aisle_embedding_size == 0 or self.hps.dept_embedding_size == 0:
          # meh.
          return
        # Log the size of gradient updates to tensorboard
        gradient_varnames = ['RNN/output_w', 'dept_embeddings', 'aisle_embeddings']
        # TODO: how to handle product embeddings? Updates should be v. sparse.
        with tf.variable_scope(tf.get_variable_scope(), reuse=True):
          gradient_vars = [tf.get_variable(vname) for vname in gradient_varnames]
          # I think compute_gradients shares its implementation with the base Optimizer
          # (i.e. these are the 'raw' gradients, and not the actual updates computed
          # by Adam with momentum etc.)
          grads = optimizer.compute_gradients(self.total_cost, var_list=gradient_vars)
          for (grad, var) in grads:
            colidx = var.name.rfind(':')
            basename = var.name[ len('instarnn/') : colidx ]
            summ_name ='Gradients/{}'.format(basename)
            self.add_summary(summ_name, grad)

          if self.hps.cell != 'lstm':
            tf.logging.warn("Histogram logging not implemented for cell {}".format(self.hps.cell))
            return
          cellvars = [tf.get_variable('rnn/LSTMCell/'+v) 
            for v in ['W_xh', 'W_hh', 'bias']
            ]
          # TODO: Ideally we could go even further and cut up the xh gradients by
          # feature (family). That'd be siiiiiiiick.
          cellgrads = optimizer.compute_gradients(self.total_cost, var_list=cellvars)
          for (grad, var) in cellgrads:
            colidx = var.name.rfind(':')
            parenidx = var.name.rfind('/')
            basename = var.name[parenidx+1:colidx]
            bygate = tf.split(grad, 4, axis=0 if basename == 'bias' else 1)
            gates = ['input_gate', 'newh_gate', 'forget_gate', 'output_gate']
            for (subgrads, gatename) in zip(bygate, gates):
              summname = 'Gradients/LSTMCell/{}/{}'.format(basename, gatename)
              self.add_summary(summname, subgrads)
        


    def _build_embedding_inputs(self, input_vars, max_seq_len):
        batch_size = self.flags.batch_size
        #max_seq_len = tf.shape(self.input_data)[1] # this should be dynamic
        embedding_dat = [
            ('pid', 'product', self.flags.prod_embed_size, N_PRODUCTS),
            ('aisleid', 'aisle', self.flags.ail_embed_size, N_AISLES),
            ('deptid', 'dept', self.flags.dep_embed_size, N_DEPARTMENTS),
        ]
        input_embeddings = []
        for (input_key, name, size, n_values) in embedding_dat:
            if size == 0:
                print('Skipping embeddings for {}'.format(name))
                continue
            input_ids = input_vars[input_key] - 1
            x = self._get_embedding(layer_name=name, inputs=input_ids,v=n_values,m=size,reuse=False) # B, M
            x = tf.reshape(x, [batch_size, 1, size])
            x = tf.tile(x, [1, max_seq_len, 1]) # B, S, M
            input_embeddings.append(x)
        return input_embeddings
