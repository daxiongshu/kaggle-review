"""
this is based on 43th solution
https://github.com/colinmorris/instacart-basket-prediction
https://www.kaggle.com/c/instacart-market-basket-analysis/discussion/38159
"""
from models.tf_models.BaseRnnModel import BaseRnnModel

class rnnModel(BaseRnnModel):

    def __init__(self,flags):
        super().__init__(flags)

    
    def _build(self):
        pass


    def _build_embedding_inputs(self, input_vars):
        embedding_dat = [
            ('pid', 'product', self.hps.product_embedding_size, N_PRODUCTS),
            ('aisleid', 'aisle', self.hps.aisle_embedding_size, N_AISLES),
            ('deptid', 'dept', self.hps.dept_embedding_size, N_DEPARTMENTS),
        ]
        input_embeddings = []
        for (input_key, name, size, n_values) in embedding_dat:
            if size == 0:
                tf.logging.info('Skipping embeddings for {}'.format(name))
                continue
            embeddings = tf.get_variable('{}_embeddings'.format(name),
                [n_values, size])
            self.add_summary(
                'Embeddings/{}_norm'.format(name), tf.norm(embeddings, axis=1)
            )
            idname = '{}_ids'.format(name)
            # TODO: Maybe everything would be simpler if the model just received
            # a monolithic input tensor, which included the already-looked-up
            # embeddings? 
            input_ids = input_vars[input_key] - 1 # go from 1-indexing to 0-indexing
            #setattr(self, idname, input_ids)
            lookuped = tf.nn.embedding_lookup(
                embeddings,
                input_ids,
                max_norm=None, # TODO: experiment with this param
            )
            lookuped = tf.reshape(lookuped, [self.batch_size, 1, size])
            lookuped = tf.tile(lookuped, [1, self.max_seq_len, 1])
            input_embeddings.append(lookuped)
        return input_embeddings
