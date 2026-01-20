from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import re
import requests
from typing import Dict, List, Any, Optional, TypedDict
from mathruler.grader import extract_boxed_content
from tqdm import tqdm
import numpy as np
import time
from collections import Counter
from sklearn.cluster import AgglomerativeClustering
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

DISPATCHER_URL = "http://localhost:8001/dispatch"

class DispatcherResponse(TypedDict):
    question: str
    majority_answer: Optional[str]
    self_consistency_score: float
    total_samples: int
    all_answers: List[Optional[str]]
    raw_responses: List[Dict[str, Any]]

def call_self_consistency_dispatcher(question: str) -> DispatcherResponse:
    """Calls the self-consistency dispatcher for a single question."""
    payload = {
        "question": question,
        "n": 5, # Sampling n=5 for efficiency in the reward function
        "max_turns": 5
    }

    if question == "":
        return {
            "question": question,
            "majority_answer": None,
            "self_consistency_score": 0.0,
            "total_samples": 0,
            "all_answers": [],
            "raw_responses": []
        }

    try:
        response = requests.post(DISPATCHER_URL, json=payload, timeout=300)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Error calling dispatcher: {e}")
        return {
            "question": question,
            "majority_answer": None,
            "self_consistency_score": 0.0,
            "total_samples": 0,
            "all_answers": [],
            "raw_responses": []
        }

def reward_self_consistency_scores(questions: List[str], max_threads: int = 5) -> List[DispatcherResponse]:
    """Fetches self-consistency scores for a batch of questions using multiple threads."""
    results = [None] * len(questions)
    
    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        # Map indices to questions to maintain order
        future_to_idx = {executor.submit(call_self_consistency_dispatcher, q): i for i, q in enumerate(questions)}
        
        for future in tqdm(as_completed(future_to_idx), total=len(questions), desc=" - Requesting self-consistency"):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                print(f"Thread error for index {idx}: {e}")
                results[idx] = {"self_consistency_score": 0.0, "majority_answer": None}
                
    return results

def reward_format(predict: str):
    pattern = re.compile(r".*<question>.*</question>.*\\boxed\{.*\}.*", re.DOTALL)
    format_match = re.fullmatch(pattern, predict)
    return 1.0 if format_match else 0.0   

def _bleu_distance_matrix(sentences):
    n = len(sentences)
    dist = np.zeros((n, n))
    smoother = SmoothingFunction().method1
    for i in tqdm(range(n), desc="  - Calculating BLEU distance matrix", leave=False):
        for j in range(i, n):
            if i == j:
                score = 1.0
            else:
                ref = [sentences[j].split()]
                hyp = sentences[i].split()
                score = sentence_bleu(ref, hyp, smoothing_function=smoother)
            dist[i, j] = dist[j, i] = 1 - score
    return dist

def penalty_cluster_share_per_problem(
        problems,
        distance_threshold: float = 0.5,
        linkage: str = "average"):
    if not problems:
        return []
    print('start clustering')
    start_time = time.time()
    dist_mat = _bleu_distance_matrix(problems)

    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=distance_threshold,
        metric="precomputed",
        linkage=linkage
    )
    labels = clustering.fit_predict(dist_mat)
    print(f'end clustering, time: {time.time() - start_time}')
    total = len(problems)
    cluster_size = Counter(labels)
    cluster_ratio = {lab: sz / total for lab, sz in cluster_size.items()}

    proportions = [cluster_ratio[lab] for lab in labels]
    return proportions


"""
    Input: predicts is a list of questions
    Output: scores is a list of scores for each question in the same order
"""
def compute_score(predicts: List[str]) -> List[Dict[str, float]]:
    """
    Computes the final reward for curriculum tasks.
    Formula: R = R_format * R_uncertainty * (1 - novelty_penalty)
    """
    results_parsing = []
    # 1. Parse predictions to extract questions
    for i in tqdm(range(len(predicts)), desc=" - Parsing predictions"):
        questions = re.findall(r"<question>(.*?)</question>", predicts[i], re.DOTALL)
        if questions:
            results_parsing.append({"question": questions[-1].strip(), "raw_response": predicts[i]})
        else:
            results_parsing.append({"question": "", "raw_response": predicts[i]})

    # 2. Get Self-Consistency scores (Uncertainty)
    questions_list = [r["question"] for r in results_parsing]
    sc_results = reward_self_consistency_scores(questions_list, max_threads=5)

    # 3. Get Novelty Penalty (Clustering)
    # We only cluster valid (non-empty) questions to avoid penalization for empty strings
    valid_questions = [q for q in questions_list if q]
    if valid_questions:
        novelty_proportions = penalty_cluster_share_per_problem(valid_questions)
        # Map back to full list
        prop_map = {q: p for q, p in zip(valid_questions, novelty_proportions)}
        full_proportions = [prop_map.get(q, 1.0) if q else 1.0 for q in questions_list]
    else:
        full_proportions = [1.0] * len(questions_list)

    final_scores = []
    for i in range(len(predicts)):
        # A. Format Reward (0 or 1)
        fmt_reward = reward_format(predicts[i])
        
        # B. Uncertainty Reward (Tent Function)
        # f(p) = 1 - |2p - 1| 
        # Peaks at p=0.5 (reward=1.0), drops to 0 at p=0 and p=1.
        sc_res = sc_results[i]
        p_x = sc_res.get("self_consistency_score", 0.0)
        uncertainty_reward = 1.0 - abs(2 * p_x - 1.0)
        
        # Special case: if p_x is exactly 0, it might be unsolveable, 
        # but the tent function handles it (reward=0). 
        # However, Agent0 sometimes sets a floor or uses informative band filtering.
        # Here we follow the logic: more consistent != better reward for CURRICULUM.
        
        # C. Novelty Reward
        # proportion is how many samples are in this cluster.
        # Higher proportion means less novel.
        novelty_reward = 1.0 - full_proportions[i]

        # Combine
        # R = Format * Uncertainty * Novelty
        overall_reward = fmt_reward * uncertainty_reward * novelty_reward

        final_scores.append({
            "overall": float(overall_reward),
            "format_score": float(fmt_reward),
            "uncertainty_score": float(uncertainty_reward),
            "novelty_score": float(novelty_reward),
            "sc_score": float(p_x)
        })

    return final_scores
