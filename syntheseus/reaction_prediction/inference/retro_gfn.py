"""Inference wrapper for the RetroGFN model.
"""
import sys
from pathlib import Path
from typing import Union, List, Sequence

from syntheseus import Molecule, SingleProductReaction, Bag
from syntheseus.reaction_prediction.inference.base import ExternalBackwardReactionModel


class RetroGFNModel(ExternalBackwardReactionModel):
    def __init__(self,
                 model_dir: Union[str, Path],
                 device: str = "cuda:0",
                 repeat: int = 10,
                 temperature: float = 0.7,
                 batch_size: int = 100,
                 gflownet_repo_root_dir: str = '../..',
                 **kwargs) -> None:
        """Initializes the RetroGFN model wrapper.
        :argument
        model_dir: str or Path. The folder should contain the following files:
            - `best_gfn.ckpt` is the GFN model checkpoint
            - `operative_config.txt` is the GFN model config.
        """
        super().__init__(model_dir, device, **kwargs)
        sys.path.append(gflownet_repo_root_dir)
        from grid_search import grid_search  # type: ignore
        from gflownet.gfns.retro.retro_gfn_single_step import RetroGFNSingleStep

        self.model = RetroGFNSingleStep(model_dir=model_dir, repeat=repeat, device=device, temperature=temperature,
                                        root_dir=gflownet_repo_root_dir)
        self.batch_size = batch_size

    def get_parameters(self):
        return []

    def _get_reactions(self, inputs: List[Molecule], num_results: int) -> List[Sequence[SingleProductReaction]]:
        outputs_list = self.model.predict([molecule.smiles for molecule in inputs], num_results,
                                          self.batch_size)
        self.model
        proper_output_list = [
            [
                SingleProductReaction(
                    reactants=Bag([Molecule(reactant.smiles) for reactant in reaction.reactants]),
                    product=Molecule(reaction.product.smiles),
                )
                for reaction in reactions
            ]
            for reactions in outputs_list

        ]
        return proper_output_list
