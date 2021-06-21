from typing import Optional

import jax
import jax.numpy as jnp
import haiku as hk

class relational_model(hk.Module):
    def __init__(self,
                 n_rels: int,
                 max_comps: int,
                 embed_dim: int,
                 name: Optional[str]=None):
        """Constructs a model with a linear layer to get log energies for links between a number of 
        components.

        Args:
            n_rels:     Number of different types of relations that can exist between any two components.
            max_comps:  The maximum number of components in any sample. All samples will be padded to 
                        these many components.
            embed_dim:  The dimension of embedding of each component.Assumes fixed size embedding for each
                        component.
        """
        super().__init__(name=name)
        self.n_rels = n_rels
        self.max_comps = max_comps
        self.embed_dim = embed_dim
        self.w = hk.Linear(self.n_rels*self.embed_dim, with_bias=False)

    def _format_log_energies(self,
                             log_energies: jnp.ndarray,
                             pad_mask: jnp.ndarray) -> jnp.ndarray:
        """
        Args:
            log_energies:   A [self.max_comps, self.max_comps, self.n_rels] sized array, having the log energy for 
                            existence of link from component i to component j of type k at the (i,j,k)-th location.
            pad_mask:       A [self.max_comps] sized array where i-th entry is 1 if the i-th component is an actual 
                            one or a component corresponding to root and 0 if it is a padded one.
        
        Returns:
            Formatted log_energies. As the 0-th component corresponds to connection to root, we need to make sure 
            log_energies for connection from 0-th component are not allowed. Moreover, in this formatted log energies
            we set the energies to and from pad components to -infinity.
        """
        
        log_energies = jax.ops.index_update(log_energies, 
                                            (0,jnp.array([i for i in range(self.max_comps)]), 
                                             jnp.array([i for i in range(self.n_rels)])), 
                                            -jnp.inf)
        
        available_from_to = jnp.logical_and(jnp.expand_dims(pad_mask, axis=-1), pad_mask)
        
        formatted_log_energies = jnp.where(available_from_to, 
                                           jnp.transpose(log_energies, (2,0,1)), 
                                           -jnp.inf)
        
        return jnp.transpose(formatted_log_energies, (1,2,0))
    
    def _call(self,
              embds: jnp.ndarray,
              choice_mask: jnp.ndarray) -> jnp.ndarray:
        """Single sample version of self.__call__(). See the same for documentation."""
        indices = jnp.where(choice_mask)
        from_embds = jnp.take_along_axis(embds, jnp.expand_dims(indices, axis=-1), axis=0)
        extra_comps_to_pad = self.M-jnp.shape(indices)[0]-1
        
        from_embds = jnp.pad(from_embds, 
                             pad_width=((1, extra_comps_to_pad), (0,0)), 
                             constant_values=((1,0), (0,0)))
        
        to_embds = jnp.reshape(self.w(from_embds), (-1, self.embed_dim, self.n_rels))
        
        log_energies = jnp.dot(from_embds, to_embds)
        
        pad_mask = jnp.array([1]*(self.M-extra_comps_to_pad)+[0]*extra_comps_to_pad)
        
        return self._format_log_energies(log_energies, pad_mask)
    
    def __call__(self,
                 embds: jnp.ndarray,
                 choice_mask: jnp.ndarray) -> jnp.ndarray:
        """
        Args:
            embds:        A [batch_size, seq_len, embed_dim] sized array having the embeddings of all the 
                          words in all sequences in a batch.
            
            choice_mask:  A [batch_size, seq_len] sized mask, which is 1 for the sequence positions which are to 
                          be included as components for relation prediction, and 0 for other positions.
        Returns:
            Log energies array of shape [batch_size, self.max_comps, self.max_comps, self.n_rels] where the (i,j,k,l)-th
            entry corresponds to the log energy of there being a link from component j to component k, of type l in the 
            i-th sample. A link to position 0, from component x indicates the log_energy of component x being the root.
        """
        return jax.vmap(self._call)(embds, choice_mask)