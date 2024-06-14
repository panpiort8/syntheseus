#!/usr/bin/env python
# coding: utf-8
import argparse
# In[56]:


import json
import os
from collections import defaultdict
from pathlib import Path
import sys
import dgl
from rdkit import Chem
from tqdm import tqdm
import torch
from typing import List


def is_molecule(smiles):
    from rdkit.Chem import MolFromSmiles
    mol = MolFromSmiles(smiles)
    if mol is None:
        return False
    Chem.RemoveHs(mol)
    return mol.GetNumAtoms() >= 1


def canonicalize_smiles(smiles):
    from rdkit.Chem import MolFromSmiles
    mol = MolFromSmiles(smiles)
    Chem.RemoveHs(mol)
    return Chem.MolToSmiles(mol)


def run_filtering(filtered_output_path: Path, results_dir: Path):
    if filtered_output_path.exists():
        return
    filtered_output_path.parent.mkdir(exist_ok=True, parents=True)

    results_path = list(results_dir.glob("*.json"))[0]
    filtered_results = json.load(open(results_path))
    reaction_dict_list = []
    for prediction_list in tqdm(filtered_results['predictions']):
        if type(prediction_list) is dict:
            prediction_list = prediction_list['predictions']
            if len(prediction_list) == 0:
                reaction_dict_list.append(
                    {'product': None, 'reactant_list_list': []})
                continue
            product = prediction_list[0]['input']['smiles']
            reactants_key = 'output'
        else:
            product = prediction_list[0]['products'][0]['smiles']
            reactants_key = 'reactants'
        if len(prediction_list) == 0:
            reaction_dict_list.append(
                {'product': None, 'reactant_list_list': []})
            continue

        product = canonicalize_smiles(product)
        reactant_list_list = []
        for prediction in prediction_list:
            reactant_list = [r['smiles'] for r in prediction[reactants_key]]
            if all(is_molecule(m) for m in reactant_list):
                reactant_list = [canonicalize_smiles(m) for m in reactant_list]
                reactant_list_list.append(reactant_list)
        reaction_dict_list.append(
            {'product': product, 'reactant_list_list': reactant_list_list})

    with open(filtered_output_path, 'w') as f:
        json.dump(reaction_dict_list, f)


def run_in_dataset_indicator(in_dataset_output_path: Path, filtered_output_path: Path, dataset: str, project_dir: Path):
    if in_dataset_output_path.exists():
        return
    in_dataset_output_path.parent.mkdir(exist_ok=True, parents=True)

    dataset_path = project_dir / f'data/{dataset}/converted/test.jsonl'
    product_to_ground_truth_reactants_set = defaultdict(set)
    for line in open(dataset_path, 'r'):
        reaction_dict = json.loads(line)
        product = canonicalize_smiles(reaction_dict['products'][0]['smiles'])
        reactants = [canonicalize_smiles(r['smiles']) for r in reaction_dict['reactants']]
        product_to_ground_truth_reactants_set[product].add(frozenset(reactants))
    product_to_ground_truth_reactants_set = dict(product_to_ground_truth_reactants_set)
    product_to_ground_truth_reactants_set[None] = set()

    filtered_results = json.load(open(filtered_output_path))
    in_dataset_indicator_list = []
    for reaction_dict in tqdm(filtered_results):
        product = reaction_dict['product']
        reactant_list_list = reaction_dict['reactant_list_list']
        in_dataset_indicator = []
        for reactant_list in reactant_list_list:
            reactant_set = frozenset(reactant_list)
            in_dataset = reactant_set in product_to_ground_truth_reactants_set[product]
            in_dataset_indicator.append(in_dataset)
        in_dataset_indicator_list.append(in_dataset_indicator)

    with open(in_dataset_output_path, 'w') as f:
        json.dump(in_dataset_indicator_list, f)


@torch.no_grad()
def compute_chemformer_feasibility(chemformer_output_path: Path, chemformer_model_dir: Path, filtered_output_path: Path,
                                   device: str, max_k: int = 50):
    if chemformer_output_path.exists():
        return
    from gflownet.gfns.retro.proxies.retro_chemformer_proxy import ReactionChemformer
    chemformer_output_path.parent.mkdir(exist_ok=True, parents=True)
    filtered_results = json.load(open(filtered_output_path))
    product_and_reactants_list = [(r['product'], r['reactant_list_list'][:max_k]) for r in filtered_results]

    model = ReactionChemformer(
        model_dir=chemformer_model_dir,
        num_results=1
    )

    model.model.model.to(device)

    feasibility_list_list = []
    for product, reactant_list_list in tqdm(product_and_reactants_list):
        if len(reactant_list_list) == 0:
            feasibility_list_list.append([])
            continue
        feasibility_list = model.forward(reactants=reactant_list_list, products=[product] * len(reactant_list_list))
        feasibility_list_list.append(feasibility_list)

    with open(chemformer_output_path, 'w') as f:
        json.dump(feasibility_list_list, f)


@torch.no_grad()
def compute_rfm_feasibility(rfm_output_path: Path, filtered_output_path: Path, project_dir: Path,
                            device: str, max_k: int = 50):
    if rfm_output_path.exists():
        return
    sys.path.append(str(project_dir / 'external/reaction_feasibility_model'))
    from rfm.models import ReactionGNN
    from rfm.featurizers import ReactionFeaturizer

    rfm_output_path.parent.mkdir(exist_ok=True, parents=True)
    filtered_results = json.load(open(filtered_output_path))
    product_and_reactants_list = [(r['product'], r['reactant_list_list'][:max_k]) for r in filtered_results]

    model = ReactionGNN(
        checkpoint_path=project_dir / 'checkpoints/feasibility_proxies/rfm/eval/best_reaction.pt',
    )
    model = model.eval()
    model = model.to(device)

    featurizer = ReactionFeaturizer()

    def get_batch(product: str, reactants_list_list: List[List[str]]):
        product_graph_list = [featurizer.featurize_smiles_single(product)] * len(reactants_list_list)
        reactants_graph_list = []
        for reactants_list in reactants_list_list:
            reactant_graphs = [featurizer.featurize_smiles_single(reactant) for reactant in reactants_list]
            reactants_graph = dgl.merge(reactant_graphs)
            reactants_graph_list.append(reactants_graph)
        return dgl.batch(product_graph_list), dgl.batch(reactants_graph_list)

    feasibility_list_list = []
    for product, reactant_list_list in tqdm(product_and_reactants_list):
        if len(reactant_list_list) == 0:
            feasibility_list_list.append([])
            continue
        product_graph, reactants_graph = get_batch(product, reactant_list_list)
        product_graph = product_graph.to(device)
        reactants_graph = reactants_graph.to(device)
        with torch.no_grad():
            feasibility_list = model(reactants=reactants_graph, products=product_graph)
            feasibility_list = torch.sigmoid(feasibility_list)
        feasibility_list = feasibility_list.cpu().numpy().tolist()
        feasibility_list_list.append(feasibility_list)

    with open(rfm_output_path, 'w') as f:
        json.dump(feasibility_list_list, f)


def report_ftc(feasibility_output_path: Path, threshold: float, k: int, in_dataset_indicator_path: Path | None = None):
    import torch
    import numpy as np
    feasibility_list_list = json.load(open(feasibility_output_path))
    if in_dataset_indicator_path:
        in_dataset_list_list = json.load(open(in_dataset_indicator_path))
    else:
        in_dataset_list_list = [[0.0] * len(feasibility) for feasibility in feasibility_list_list]
    counts = []
    for feasibility_list, in_dataset_list in zip(feasibility_list_list, in_dataset_list_list):
        if len(feasibility_list) == 0:
            counts.append(0)
        else:
            feasibilities = torch.tensor(feasibility_list[:k])
            in_dataset_indicators = torch.tensor(in_dataset_list[:k])
            feasibilities = feasibilities + in_dataset_indicators
            count = (feasibilities > threshold).sum().item() / k
            counts.append(count)
    return np.mean(counts), np.std(counts)


if __name__ == '__main__':
    import warnings

    warnings.filterwarnings('ignore')

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True, choices=['uspto_50k', 'uspto_mit'])
    parser.add_argument('--project_dir', type=str, default='.', help='The relative path to the project root directory')
    parser.add_argument('--results_dir', type=str, required=True,
                        help='The relative path to the results directory that was dumped by the Syntheseus framework')
    parser.add_argument('--metrics', type=str, nargs='+', help='The metrics to compute',
                        default=['acc', 'round_trip', 'ftc'])

    args = parser.parse_args()
    device = args.device
    dataset = args.dataset
    project_dir = Path(args.project_dir)
    results_dir = Path(args.results_dir)
    metrics = args.metrics

    model = str(results_dir).split('/')[-1]
    eval_data_dir = project_dir / 'experiments' / 'evaluation'
    metrics_output_path = eval_data_dir / 'metrics' / dataset / f'{model}.json'

    metric_dict = {}
    k_list = [1, 3, 5, 10, 20, 50]
    metric_dict['k'] = k_list

    if 'acc' in metrics:
        results_path = list(results_dir.glob("*.json"))[0]
        results = json.load(open(results_path))
        metric_dict['top_k'] = [round(results['chosen_top_k'][str(k)] * 100, 2) for k in k_list]
        metric_dict['mrr'] = round(results['mrr'], 4)

    filtered_output_path = eval_data_dir / 'filtered' / dataset / f'{model}.json'
    in_dataset_output_path = eval_data_dir / 'in_dataset_indicator' / dataset / f'{model}.json'

    run_filtering(filtered_output_path, results_dir)
    run_in_dataset_indicator(in_dataset_output_path, filtered_output_path, dataset, project_dir)

    if 'round_trip' in metrics:
        chemformer_output_path = eval_data_dir / 'feasibility_chemformer' / dataset / f'{model}.json'
        chemformer_model_dir = project_dir / 'checkpoints/feasibility_proxies/chemformer/eval'
        compute_chemformer_feasibility(chemformer_output_path, chemformer_model_dir, filtered_output_path, device)
        round_trip_list = []
        for k in k_list:
            ftc = report_ftc(chemformer_output_path, threshold=0.0, k=k)
            round_trip_list.append((round(ftc[0] * 100, 1), round(ftc[1] * 100, 1)))
        metric_dict['round_trip'] = round_trip_list

    if 'ftc' in metrics:
        rfm_output_path = eval_data_dir / 'feasibility_rfm' / dataset / f'{model}.json'
        ftc_checkpoint_path = project_dir / 'checkpoints/feasibility_proxies/rfm/eval/best_reaction.pt'
        compute_rfm_feasibility(rfm_output_path, filtered_output_path, project_dir, device)
        for threshold in [0.7, 0.8, 0.9]:
            ftc_k_list = []
            for k in k_list:
                ftc = report_ftc(rfm_output_path, threshold=threshold, k=k,
                                 in_dataset_indicator_path=in_dataset_output_path)
                ftc_k_list.append((round(ftc[0] * 100, 1), round(ftc[1] * 100, 1)))
            metric_dict[f"ftc_{threshold}"] = ftc_k_list

    metric_dict_json = json.dumps(metric_dict)
    metrics_output_path.parent.mkdir(exist_ok=True, parents=True)
    with open(metrics_output_path, 'w') as f:
        f.write(metric_dict_json)
